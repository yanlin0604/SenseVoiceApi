import io
import asyncio
import time
import numpy as np
import torch
from loguru import logger
from models.engine_loader import engine_loader
from services.webhook_client import webhook_client
from core.text_postprocess import normalize_asr_text
from config import config

class AudioPipeline:
    def __init__(self, session_id: str, mode: int = 1):
        """
        :param session_id: 当前连接的会话ID
        :param mode: 1 为纯语音识别，2 为说话人分离
        """
        self.session_id = session_id
        self.mode = mode
        
        # 音频块缓冲
        self.audio_buffer = np.array([], dtype=np.float32)
        # VAD 积累的语音段
        self.audio_vad = np.array([], dtype=np.float32)
        
        # VAD 相关状态
        self.vad_cache = {}
        self.last_vad_beg = -1
        self.last_vad_end = -1
        self.offset = 0
        
        # VAD param & 内部切片大小
        # 优化：240ms 兼顾了低延迟（秒出字）和低 CPU 开销（每秒仅 4 次推理），并且是 60ms 的整数倍
        self.chunk_size_ms = 240
        # 计算 16kHz 下需要的采样点数
        self.chunk_size_samples = int(16000 * self.chunk_size_ms / 1000)
        
        # 会话级临时声纹存储 { speaker_name: numpy_array_embedding }
        self.speaker_embeddings = {}
        self.speaker_counter = 0
        self.sv_threshold = config.sv_similarity_threshold
        
        # 原始二进制缓冲
        self.byte_buffer = bytearray()
        
        # ========== 流式 2pass 相关状态 ==========
        self.streaming_enabled = config.enable_streaming and engine_loader.streaming_model is not None
        self.is_2pass = config.is_2pass_mode
        
        if self.streaming_enabled:
            # Paraformer-zh-streaming 的流式推理缓存
            self.streaming_cache = {}
            # 流式 chunk 参数
            self.streaming_chunk_size = config.parsed_chunk_size  # e.g. [0, 10, 5]
            self.encoder_chunk_look_back = config.streaming_encoder_chunk_look_back
            self.decoder_chunk_look_back = config.streaming_decoder_chunk_look_back
            # 用于精修阶段的完整音频段缓冲
            self.segment_audio_for_refine = np.array([], dtype=np.float32)
            # 最后一次流式识别的文本（纯流式模式用作最终结果）
            self._last_streaming_text = ""

            # ===== 流式喂块解耦 =====
            # paraformer-streaming 要求每次喂入 = chunk_size[1] × 960 采样点
            # VAD 用 240ms 切块（断句灵敏），流式需累积到这个步长才推一次，否则吞字/错位
            self.streaming_feed_samples = int(self.streaming_chunk_size[1] * 960)
            # 流式专用累积缓冲（与 VAD 解耦）
            self.streaming_pending = np.array([], dtype=np.float32)

            logger.info(f"[{session_id}] 流式模式已启用, 2pass={'是' if self.is_2pass else '否'}, "
                        f"chunk_size={self.streaming_chunk_size}, 流式喂块={self.streaming_feed_samples}采样点"
                        f"({self.streaming_feed_samples/16}ms)")
        else:
            logger.info(f"[{session_id}] 离线模式 (流式未启用)")

    async def process_chunk(self, audio_bytes: bytes, websocket=None):
        """
        完全还原原版 server_wss_v3_new.py 的流式 VAD 逻辑。
        新增流式推理分支：当启用 streaming 时，每个 chunk 同时送入流式模型产出中间结果。
        """
        self.byte_buffer.extend(audio_bytes)
        
        # 保证解析 16bit 时字节数是偶数
        usable_bytes = len(self.byte_buffer) - (len(self.byte_buffer) % 2)
        if usable_bytes > 0:
            chunk_data = self.byte_buffer[:usable_bytes]
            self.byte_buffer = self.byte_buffer[usable_bytes:]
            
            # 转为 float32
            samples = np.frombuffer(chunk_data, dtype=np.int16).astype(np.float32) / 32768.0
            self.audio_buffer = np.append(self.audio_buffer, samples)
            
            # 按 chunk_size 分割给 VAD
            while len(self.audio_buffer) >= self.chunk_size_samples:
                chunk = self.audio_buffer[:self.chunk_size_samples]
                self.audio_buffer = self.audio_buffer[self.chunk_size_samples:]
                
                self.audio_vad = np.append(self.audio_vad, chunk)
                
                # 同时缓存音频供精修阶段使用
                if self.streaming_enabled:
                    self.segment_audio_for_refine = np.append(self.segment_audio_for_refine, chunk)
                
                try:
                    # 使用 asyncio.gather 让 VAD 和流式 ASR 并发执行
                    vad_task = asyncio.to_thread(
                        engine_loader.vad_model.generate,
                        input=chunk,
                        cache=self.vad_cache,
                        is_final=False,
                        chunk_size=self.chunk_size_ms,
                    )
                    
                    if self.streaming_enabled:
                        streaming_task = self._process_streaming_chunk(chunk, websocket)
                        res, _ = await asyncio.gather(vad_task, streaming_task)
                    else:
                        res = await vad_task
                    
                    if len(res) > 0 and len(res[0].get("value", [])) > 0:
                        vad_segments = res[0]["value"]
                        for segment in vad_segments:
                            if segment[0] > -1:  # speech begin
                                self.last_vad_beg = segment[0]
                                
                            if segment[1] > -1:  # speech end
                                self.last_vad_end = segment[1]
                                
                            # 凑齐了一个完整的说话段落
                            if self.last_vad_beg > -1 and self.last_vad_end > -1:
                                self.last_vad_beg -= self.offset
                                self.last_vad_end -= self.offset
                                self.offset += self.last_vad_end
                                
                                beg = int(self.last_vad_beg * 16000 / 1000)
                                end = int(self.last_vad_end * 16000 / 1000)
                                
                                segment_audio = self.audio_vad[beg:end]
                                
                                if len(segment_audio) > 0:
                                    logger.info(f"[{self.session_id}] VAD 成功分割出语音段，长度: {len(segment_audio)} 采样点")
                                    await self.flush_segment(segment_audio, websocket)
                                    
                                # 截断处理过的音频
                                if end <= len(self.audio_vad):
                                    self.audio_vad = self.audio_vad[end:]
                                else:
                                    self.audio_vad = np.array([], dtype=np.float32)
                                    
                                self.last_vad_beg = -1
                                self.last_vad_end = -1
                                
                    # 内存防爆处理：如果长时间没说话（超过5秒的静音），截断前面没用的静音和缓存，防止内存爆炸和推理卡死
                    if self.last_vad_beg == -1 and len(self.audio_vad) > 16000 * 5:
                        # 扔掉前面的数据，只保留最后 1 秒的音频防止截断刚开始的语音
                        drop_samples = len(self.audio_vad) - 16000
                        self.audio_vad = self.audio_vad[drop_samples:]
                        self.offset += (drop_samples / 16.0)  # offset 是毫秒
                        logger.debug(f"[{self.session_id}] 丢弃超长静音 {drop_samples} 采样点，更新 offset={self.offset}")
                        
                        # 同步清空流式模型的无限增长缓存，否则超过 20 秒静音会导致算力爆炸并断开 WebSocket
                        if self.streaming_enabled:
                            self.streaming_cache = {}
                            self.streaming_pending = np.array([], dtype=np.float32)
                            # 保留最后 1 秒对应的数据给精修阶段
                            if len(self.segment_audio_for_refine) > 16000:
                                self.segment_audio_for_refine = self.segment_audio_for_refine[-16000:]
                            self._last_streaming_text = ""
                                
                except Exception as e:
                    logger.error(f"VAD/流式推理出错: {e}")

    async def _process_streaming_chunk(self, chunk: np.ndarray, websocket=None):
        """
        将音频送入 Paraformer-zh-streaming 进行流式推理。

        关键修复：paraformer-streaming 要求每次喂入 = chunk_size[1]×960 采样点。
        VAD 用 240ms 切块，这里先累积到流式步长（如 480ms）再整块推理，
        避免半截块导致 encoder/decoder 缓存错位、吞字、滞后。
        """
        try:
            # 先把 VAD 的小块累积进流式专用缓冲
            self.streaming_pending = np.append(self.streaming_pending, chunk)

            # 没攒够一个完整步长，先不推理（直接返回，等下一个 VAD 块）
            if len(self.streaming_pending) < self.streaming_feed_samples:
                return

            # 取出整数倍步长喂给模型，余下的留到下次
            feed_len = (len(self.streaming_pending) // self.streaming_feed_samples) * self.streaming_feed_samples
            feed_audio = self.streaming_pending[:feed_len]
            self.streaming_pending = self.streaming_pending[feed_len:]

            t0 = time.monotonic()
            streaming_res = await asyncio.to_thread(
                engine_loader.streaming_model.generate,
                input=feed_audio,
                cache=self.streaming_cache,
                is_final=False,
                chunk_size=self.streaming_chunk_size,
                encoder_chunk_look_back=self.encoder_chunk_look_back,
                decoder_chunk_look_back=self.decoder_chunk_look_back,
            )
            cost_ms = (time.monotonic() - t0) * 1000

            if streaming_res and len(streaming_res) > 0 and streaming_res[0].get("text"):
                chunk_text = streaming_res[0]["text"]
                if chunk_text:
                    self._last_streaming_text += chunk_text
                    # paraformer 流式输出无 <|zh|> 标签，无需每块清洗全文；直接推累加结果
                    await self._send_result(
                        websocket, text=self._last_streaming_text, is_final=False, speaker=None
                    )
            # 耗时埋点：流式单次推理 ms（喂入时长 feed_len/16 ms）
            logger.debug(f"[{self.session_id}] ⏱ 流式推理 {cost_ms:.0f}ms "
                         f"(喂入 {feed_len/16:.0f}ms 音频)")
        except Exception as e:
            logger.error(f"[{self.session_id}] 流式推理出错: {e}")

    async def flush_segment(self, segment_audio: np.ndarray, websocket=None):
        """
        对已截断的语音段进行精修 ASR 与声纹识别。
        
        2pass 模式：用精修模型重新转写完整音频段 → 替换前端流式中间结果
        纯流式模式：用最后一次流式结果作为最终结果
        离线模式：直接用精修模型识别（兼容原逻辑）
        """
        try:
            seg_ms = len(segment_audio) / 16.0  # 音频段时长(ms)
            t_start = time.monotonic()
            # ===== 1. 文本识别 =====
            text = ""
            t_asr = time.monotonic()

            if self.streaming_enabled and not self.is_2pass:
                # 纯流式模式（精修模型 == 流式模型）：
                # 用 is_final=True 做最后一次流式推理获取最终文本
                try:
                    final_res = await asyncio.to_thread(
                        engine_loader.streaming_model.generate,
                        input=segment_audio,
                        cache=self.streaming_cache,
                        is_final=True,
                        chunk_size=self.streaming_chunk_size,
                        encoder_chunk_look_back=self.encoder_chunk_look_back,
                        decoder_chunk_look_back=self.decoder_chunk_look_back,
                    )
                    if final_res and len(final_res) > 0 and final_res[0].get("text"):
                        text = final_res[0]["text"]
                    else:
                        text = self._last_streaming_text
                except Exception as e:
                    logger.warning(f"[{self.session_id}] 流式 final 推理失败，使用最后缓存文本: {e}")
                    text = self._last_streaming_text

                # 声纹识别（mode2）与文本无依赖，纯流式时文本已就绪，直接顺序取
                speaker = None
                if self.mode == 2:
                    t_sv = time.monotonic()
                    speaker = await self._identify_speaker(segment_audio)
                    sv_cost = (time.monotonic() - t_sv) * 1000
                else:
                    sv_cost = 0
                asr_cost = (time.monotonic() - t_asr) * 1000
            else:
                # 2pass / 离线模式：精修 ASR 与声纹识别互不依赖，并行执行省时间
                async def _run_asr():
                    asr_res = await asyncio.to_thread(
                        engine_loader.asr_model.generate, input=segment_audio
                    )
                    if asr_res and len(asr_res) > 0 and asr_res[0].get("text"):
                        return asr_res[0]["text"]
                    return ""

                if self.mode == 2:
                    t_par = time.monotonic()
                    text, speaker = await asyncio.gather(
                        _run_asr(), self._identify_speaker(segment_audio)
                    )
                    asr_cost = sv_cost = (time.monotonic() - t_par) * 1000  # 并行，合计墙钟时间
                else:
                    text = await _run_asr()
                    speaker = None
                    asr_cost = (time.monotonic() - t_asr) * 1000
                    sv_cost = 0

            # ===== 2. 文本后处理：清洗标签 + 数字归一化 + 标点补全 =====
            punc_cost = 0
            if text:
                t_punc = time.monotonic()
                punc_model = engine_loader.punc_model if config.enable_punc else None
                text = await asyncio.to_thread(normalize_asr_text, text, punc_model)
                punc_cost = (time.monotonic() - t_punc) * 1000

            if not text:
                # 如果因为是噪音导致最终识别为空，必须重置流式状态，并通知前端清理残留的流式假阳性文字
                if self.streaming_enabled:
                    self.streaming_cache = {}
                    self.streaming_pending = np.array([], dtype=np.float32)
                    self.segment_audio_for_refine = np.array([], dtype=np.float32)
                    self._last_streaming_text = ""
                await self._send_result(websocket, text="", is_final=True, speaker=speaker)
                return

            total_cost = (time.monotonic() - t_start) * 1000
            logger.info(f"[{self.session_id}] {'精修' if self.is_2pass else ''}识别结果: {text}")
            # 全链路耗时埋点：定位"越来越慢"到底卡在哪一环
            logger.info(f"[{self.session_id}] ⏱ flush耗时 总{total_cost:.0f}ms "
                        f"| ASR{asr_cost:.0f}ms 声纹{sv_cost:.0f}ms 标点{punc_cost:.0f}ms "
                        f"| 音频段{seg_ms:.0f}ms 已知说话人{len(self.speaker_embeddings)}人")

            # ===== 3. 推送最终结果（替换前端的流式中间结果） =====
            await self._send_result(
                websocket, text=text, is_final=True, speaker=speaker
            )

            # ===== 4. 推送至 Java Webhook =====
            await webhook_client.push_to_java(
                session_id=self.session_id,
                text=text,
                speaker=speaker,
                is_final=True
            )

            # ===== 5. 重置流式状态 =====
            if self.streaming_enabled:
                self.streaming_cache = {}
                self.streaming_pending = np.array([], dtype=np.float32)
                self.segment_audio_for_refine = np.array([], dtype=np.float32)
                self._last_streaming_text = ""

        except Exception as e:
            logger.error(f"处理音频段时发生错误: {e}")

    async def _identify_speaker(self, segment_audio: np.ndarray) -> str:
        """
        声纹识别：提取音频段的 embedding 并与已知说话人进行余弦相似度比对。
        """
        if not engine_loader.sv_pipeline or not hasattr(engine_loader.sv_pipeline, 'model'):
            return "未知用户"
        
        try:
            # 准备 Tensor 输入给 PyTorch 模型
            tensor_audio = torch.from_numpy(segment_audio).unsqueeze(0)
            # 如果模型在 GPU 上，移动 tensor 到对应的 device
            if hasattr(engine_loader.sv_pipeline.model, 'device'):
                tensor_audio = tensor_audio.to(engine_loader.sv_pipeline.model.device)
                
            # 直接调用底层模型的 forward 方法提取 embedding
            sv_res = await asyncio.to_thread(
                engine_loader.sv_pipeline.model,
                tensor_audio
            )
            
            emb = None
            if isinstance(sv_res, torch.Tensor):
                emb = sv_res.cpu().numpy()
            elif isinstance(sv_res, np.ndarray):
                emb = sv_res
            elif isinstance(sv_res, list) and len(sv_res) > 0:
                emb = sv_res[0].cpu().numpy() if isinstance(sv_res[0], torch.Tensor) else sv_res[0]
                
            if emb is not None:
                emb_flat = np.array(emb).flatten()
                norm1 = np.linalg.norm(emb_flat)
                
                best_spk = None
                best_score = -1.0
                
                if norm1 > 0:
                    # 与当前会话中已经记录的所有人进行余弦相似度对比
                    for spk_id, stored_emb in self.speaker_embeddings.items():
                        norm2 = np.linalg.norm(stored_emb)
                        if norm2 > 0:
                            score = np.dot(emb_flat, stored_emb) / (norm1 * norm2)
                            if score > best_score:
                                best_score = float(score)
                                best_spk = spk_id
                                
                    # 大于阈值认为是同一个人，否则新建角色
                    if best_spk and best_score >= self.sv_threshold:
                        speaker = best_spk
                        logger.info(f"[{self.session_id}] 匹配到已有说话人: {speaker} (相似度: {best_score:.3f})")
                    else:
                        self.speaker_counter += 1
                        speaker = f"用户{self.speaker_counter}"
                        self.speaker_embeddings[speaker] = emb_flat
                        logger.info(f"[{self.session_id}] 创建新说话人: {speaker} (最高相似度: {best_score:.3f})")
                    
                    return speaker
            else:
                # 打印出真实的返回结构以便调试
                logger.warning(f"SV Pipeline 未找到特征字段。实际返回结构: {type(sv_res)}, 内容: {str(sv_res)[:200]}")
                
        except Exception as e:
            logger.warning(f"SV 特征提取或比对失败: {e}")
        
        return "未知用户"

    async def _send_result(self, websocket, text: str, is_final: bool, speaker: str = None):
        """
        统一的 WebSocket 消息发送方法。
        
        消息格式：
        - is_final=false: 流式中间结果（前端应实时显示并持续替换）
        - is_final=true:  最终精修结果（前端应替换当前流式中间结果为最终文本）
        """
        if websocket:
            try:
                await websocket.send_json({
                    "session_id": self.session_id,
                    "text": text,
                    "speaker": speaker,
                    "is_final": is_final,
                    "mode": "refined" if is_final else "streaming",
                })
            except Exception as ws_err:
                logger.debug(f"回传 WebSocket 失败: {ws_err}")

    async def flush(self, websocket=None):
        """
        前端手动点击 Flush 或断开连接时调用，强制处理当前剩余的音频。
        """
        if len(self.audio_vad) > 0:
            logger.info(f"[{self.session_id}] 手动 Flush 触发，处理剩余 {len(self.audio_vad)} 采样点")
            await self.flush_segment(self.audio_vad, websocket)
            self.audio_vad = np.array([], dtype=np.float32)
            self.vad_cache = {}
            self.last_vad_beg = -1
            self.last_vad_end = -1
            self.offset = 0
