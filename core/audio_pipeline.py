import io
import asyncio
import time
import numpy as np
import torch
from loguru import logger
from models.engine_loader import engine_loader
from services.webhook_client import webhook_client
from services.speaker_service import get_speaker_info_from_api, get_candidate_feature_ids
from routers.voiceprint import get_voiceprint_backend
from modules.text_formatter import format_str_v3
from config import config
from core.concurrency import limit_vad, limit_asr, limit_sv
from typing import Optional

class AudioPipeline:
    def __init__(self, session_id: str, mode: int = 1, user_id: Optional[str] = None):
        """
        :param session_id: 当前连接的会话ID
        :param mode: 1 为纯语音识别，2 为会话内临时声纹盲分，3 为持久化声纹库实名识别+档案联动
        :param user_id: 用户唯一标识（传入时自动拉取该用户关联的候选人声纹进行定向优化）
        """
        self.session_id = session_id
        self.mode = mode
        self.user_id = user_id
        
        # 定向候选人声纹列表 [{"feature_id": "xxx", "user_id": "yyy"}, ...]
        self.candidate_feature_ids = None
        self.candidate_fetched = False
        
        # 2pass 即时流式出字相关状态
        self.last_partial_time = 0.0
        self.last_partial_text = ""
        
        # 音频块缓冲
        self.audio_buffer = np.array([], dtype=np.float32)
        # VAD 积累的语音段
        self.audio_vad = np.array([], dtype=np.float32)
        
        # VAD 相关状态
        self.vad_cache = {}
        self.last_vad_beg = -1
        self.last_vad_end = -1
        self.offset = 0
        
        # 每次喂给 VAD 的 chunk_size (来自配置，如 200ms)
        self.chunk_size_ms = config.chunk_size_ms if hasattr(config, 'chunk_size_ms') else 200
        # 计算 16kHz 下需要的采样点数
        self.chunk_size_samples = int(16000 * self.chunk_size_ms / 1000)
        
        # 会话级临时声纹存储 { speaker_name: numpy_array_embedding }
        self.speaker_embeddings = {}
        self.speaker_counter = 0
        self.sv_threshold = config.sv_similarity_threshold
        
        # 原始二进制缓冲
        self.byte_buffer = bytearray()

    async def process_chunk(self, audio_bytes: bytes, websocket=None):
        """
        完全还原原版流式 VAD 逻辑，并在说话过程中触发 2pass 即时流式出字 (边说边出字)。
        """
        # 如果是模式3且提供了 user_id，首次接收音频时异步拉取定向候选人声纹特征
        if self.mode == 3 and self.user_id and not self.candidate_fetched:
            self.candidate_fetched = True
            try:
                self.candidate_feature_ids = await get_candidate_feature_ids(self.user_id)
            except Exception as e:
                logger.warning(f"[{self.session_id}] 拉取候选声纹异常: {e}")

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
                
                try:
                    # 使用 funasr 真正的流式 VAD 接口 (通过信号量并发限流 + 线程池防阻塞)
                    async with limit_vad():
                        res = await asyncio.to_thread(
                            engine_loader.vad_model.generate,
                            input=chunk,
                            cache=self.vad_cache,
                            is_final=False,
                            chunk_size=self.chunk_size_ms,
                        )
                    
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
                                self.last_partial_text = ""
                                self.last_partial_time = 0.0

                    # 2pass 即时流式出字 (边说边出字)
                    # 当语音正在进行中 (speech begin 已触发且未触发 end)
                    if self.last_vad_beg > -1 and self.last_vad_end == -1:
                        now = time.time()
                        if now - self.last_partial_time >= 0.28:
                            beg_sample = max(0, int((self.last_vad_beg - self.offset) * 16000 / 1000))
                            current_speech = self.audio_vad[beg_sample:]
                            if len(current_speech) >= 3200:  # 至少 200ms
                                try:
                                    if config.enable_streaming and engine_loader.streaming_model:
                                        async with limit_asr():
                                            online_res = await asyncio.to_thread(
                                                engine_loader.streaming_model.generate,
                                                input=current_speech,
                                                is_final=False,
                                            )
                                    else:
                                        gen_kwargs = {
                                            "language": config.asr_language,
                                            "use_itn": config.enable_itn if config.enable_itn is not None else config.asr_use_itn,
                                            "ban_emo_unk": config.asr_ban_emo_unk,
                                        }
                                        async with limit_asr():
                                            online_res = await asyncio.to_thread(
                                                engine_loader.asr_model.generate,
                                                input=current_speech,
                                                **gen_kwargs
                                            )
                                    if online_res and online_res[0].get("text"):
                                        p_text = format_str_v3(online_res[0]["text"]).strip()
                                        if p_text and p_text != self.last_partial_text:
                                            self.last_partial_text = p_text
                                            self.last_partial_time = now
                                            if websocket:
                                                await websocket.send_json({
                                                    "session_id": self.session_id,
                                                    "text": p_text,
                                                    "is_final": False,
                                                    "mode": "2pass-online"
                                                })
                                except Exception as err:
                                    logger.debug(f"[{self.session_id}] 2pass 流式中间识别异常: {err}")
                                
                    # 内存防爆处理：如果长时间没说话（超过5秒的静音），截断前面没用的静音
                    if self.last_vad_beg == -1 and len(self.audio_vad) > 16000 * 5:
                        # 扔掉前面的数据，只保留最后 1 秒的音频防止截断刚开始的语音
                        drop_samples = len(self.audio_vad) - 16000
                        self.audio_vad = self.audio_vad[drop_samples:]
                        self.offset += (drop_samples / 16.0)  # offset 是毫秒
                        logger.debug(f"[{self.session_id}] 丢弃超长静音 {drop_samples} 采样点，更新 offset={self.offset}")
                                
                except Exception as e:
                    logger.error(f"VAD 推理出错: {e}")


    async def flush_segment(self, segment_audio: np.ndarray, websocket=None):
        """
        对已截断的语音段进行 ASR 与声纹识别
        """
        try:
            # 1. 语音识别 (SenseVoice / 离线精修 ASR) 信号量限流 + 线程池
            use_itn = config.enable_itn if config.enable_itn is not None else config.asr_use_itn
            gen_kwargs = {
                "language": config.asr_language,
                "use_itn": use_itn,
                "ban_emo_unk": config.asr_ban_emo_unk,
            }
            if config.asr_hotwords:
                gen_kwargs["postprocess_hotwords"] = config.asr_hotwords

            async with limit_asr():
                asr_res = await asyncio.to_thread(
                    engine_loader.asr_model.generate,
                    input=segment_audio,
                    **gen_kwargs
                )
            
            if not asr_res or not asr_res[0].get("text"):
                return
                
            text = format_str_v3(asr_res[0]["text"])

            # 标点符号恢复模型 (如果启用)
            if config.enable_punc and engine_loader.punc_model and text.strip():
                try:
                    punc_res = await asyncio.to_thread(
                        engine_loader.punc_model.generate,
                        input=text
                    )
                    if punc_res and len(punc_res) > 0 and punc_res[0].get("text"):
                        text = punc_res[0]["text"]
                except Exception as p_err:
                    logger.debug(f"标点恢复处理异常: {p_err}")

            if not text.strip():
                return
            logger.info(f"[{self.session_id}] 识别结果: {text}")
            
            # 2. 说话人识别与分离逻辑
            speaker = None
            speaker_name = None
            role = None
            avatar_url = None

            # [Mode 2]: 原生会话内临时声纹盲分 (保持原逻辑不变)
            if self.mode == 2:
                if engine_loader.sv_pipeline and hasattr(engine_loader.sv_pipeline, 'model'):
                    try:
                        tensor_audio = torch.from_numpy(segment_audio).unsqueeze(0)
                        if hasattr(engine_loader.sv_pipeline.model, 'device'):
                            tensor_audio = tensor_audio.to(engine_loader.sv_pipeline.model.device)
                            
                        async with limit_sv():
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
                                for spk_id, stored_emb in self.speaker_embeddings.items():
                                    norm2 = np.linalg.norm(stored_emb)
                                    if norm2 > 0:
                                        score = np.dot(emb_flat, stored_emb) / (norm1 * norm2)
                                        if score > best_score:
                                            best_score = float(score)
                                            best_spk = spk_id
                                            
                                if best_spk and best_score >= self.sv_threshold:
                                    speaker = best_spk
                                    logger.info(f"[{self.session_id}] 匹配到已有说话人: {speaker} (相似度: {best_score:.3f})")
                                else:
                                    self.speaker_counter += 1
                                    speaker = f"用户{self.speaker_counter}"
                                    self.speaker_embeddings[speaker] = emb_flat
                                    logger.info(f"[{self.session_id}] 创建新说话人: {speaker} (最高相似度: {best_score:.3f})")
                            speaker_name = speaker
                        else:
                            speaker = "未知用户"
                            speaker_name = speaker
                    except Exception as e:
                        logger.warning(f"SV 特征提取或比对失败: {e}")

            # [Mode 3]: 注册声纹库实名识别 + 第三方资料查询 (支持定向候选人优化，未命中回退为临时说话人)
            elif self.mode == 3:
                try:
                    # 1. 转换为 16k16bit mono PCM
                    pcm_bytes = (np.clip(segment_audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                    
                    # 2. 查询本地已注册声纹库 (带定向候选人过滤)
                    _, vp_client = get_voiceprint_backend()
                    top_match = None
                    target_group_id = config.default_group_id

                    async with limit_sv():
                        res = await asyncio.to_thread(
                            vp_client.search,
                            group_id=target_group_id,
                            audio_data=pcm_bytes,
                            top_k=1,
                            candidate_feature_ids=self.candidate_feature_ids
                        )
                    if res and not res.get("error") and res.get("scoreList"):
                        first = res["scoreList"][0]
                        threshold = float(getattr(vp_client, "threshold", config.local_voiceprint_threshold))
                        if float(first.get("score", 0.0)) >= threshold:
                            top_match = first

                    # 3. 如果命中已注册声纹
                    if top_match:
                        speaker = top_match.get("featureId")
                        score = float(top_match.get("score", 0.0))
                        logger.info(f"[{self.session_id}] [Mode 3] 命中已注册声纹: {speaker}, 相似度={score:.3f}")

                        # 查询第三方用户详细资料
                        spk_info = await get_speaker_info_from_api(speaker)
                        if spk_info:
                            speaker_name = spk_info.get("speaker_name") or speaker
                            role = spk_info.get("role") or ""
                            avatar_url = spk_info.get("avatar_url") or ""
                        else:
                            speaker_name = speaker
                    else:
                        # 未命中声纹库：回退为会话内临时说话人盲分
                        logger.debug(f"[{self.session_id}] [Mode 3] 声纹库未匹配，回退到临时盲分")
                        if engine_loader.sv_pipeline and hasattr(engine_loader.sv_pipeline, 'model'):
                            tensor_audio = torch.from_numpy(segment_audio).unsqueeze(0)
                            if hasattr(engine_loader.sv_pipeline.model, 'device'):
                                tensor_audio = tensor_audio.to(engine_loader.sv_pipeline.model.device)
                            async with limit_sv():
                                sv_res = await asyncio.to_thread(engine_loader.sv_pipeline.model, tensor_audio)
                            emb = sv_res.cpu().numpy() if isinstance(sv_res, torch.Tensor) else sv_res
                            if emb is not None:
                                emb_flat = np.array(emb).flatten()
                                norm1 = np.linalg.norm(emb_flat)
                                best_spk, best_score = None, -1.0
                                if norm1 > 0:
                                    for spk_id, stored_emb in self.speaker_embeddings.items():
                                        norm2 = np.linalg.norm(stored_emb)
                                        if norm2 > 0:
                                            sc = np.dot(emb_flat, stored_emb) / (norm1 * norm2)
                                            if sc > best_score:
                                                best_score = float(sc)
                                                best_spk = spk_id
                                    if best_spk and best_score >= self.sv_threshold:
                                        speaker = best_spk
                                    else:
                                        self.speaker_counter += 1
                                        speaker = f"临时用户{self.speaker_counter}"
                                        self.speaker_embeddings[speaker] = emb_flat
                                speaker_name = speaker
                                role = "临时说话人"
                            else:
                                speaker = "未知用户"
                                speaker_name = speaker
                        else:
                            speaker = "未知用户"
                            speaker_name = speaker

                except Exception as e:
                    logger.warning(f"[Mode 3] 声纹检索与资料查询异常: {e}")
                    speaker = "未知用户"
                    speaker_name = speaker

            # 3. 推送至 Java Webhook
            await webhook_client.push_to_java(
                session_id=self.session_id,
                text=text,
                speaker=speaker,
                speaker_name=speaker_name,
                role=role,
                avatar_url=avatar_url,
                is_final=True
            )
            
            # 4. (方便前端测试与统一2pass协议) 将最终定稿回传给 WebSocket 客户端
            if websocket:
                try:
                    resp_data = {
                        "session_id": self.session_id,
                        "text": text,
                        "speaker": speaker,
                        "is_final": True,
                        "mode": "2pass-offline"
                    }
                    if speaker_name:
                        resp_data["speaker_name"] = speaker_name
                    if role:
                        resp_data["role"] = role
                    if avatar_url:
                        resp_data["avatar_url"] = avatar_url

                    await websocket.send_json(resp_data)
                except Exception as ws_err:
                    logger.debug(f"回传WebSocket失败: {ws_err}")
                    
        except Exception as e:
            logger.error(f"处理音频段时发生错误: {e}")

    async def flush(self, websocket=None):
        """
        前端手动点击 Flush 或客户端断开连接时调用，强制处理当前剩余的有效音频。
        """
        if len(self.audio_vad) > 0:
            beg_sample = 0
            if self.last_vad_beg > -1:
                beg_sample = max(0, int((self.last_vad_beg - self.offset) * 16000 / 1000))
            remaining = self.audio_vad[beg_sample:] if beg_sample < len(self.audio_vad) else self.audio_vad
            if len(remaining) >= 3200:
                logger.info(f"[{self.session_id}] 手动/结束 Flush 触发，处理剩余 {len(remaining)} 采样点")
                await self.flush_segment(remaining, websocket)
            self.audio_vad = np.array([], dtype=np.float32)
            self.vad_cache = {}
            self.last_vad_beg = -1
            self.last_vad_end = -1
            self.offset = 0
            self.last_partial_text = ""
            self.last_partial_time = 0.0

