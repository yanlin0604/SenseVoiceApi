"""
本地声纹识别客户端模块
基于 ModelScope ERes2Net 模型实现声纹注册、验证、删除等功能
"""

import os
import json
import uuid
import tempfile
import threading
import numpy as np
import wave
from datetime import datetime
from typing import Optional, Union, List, Dict, Any
import torch
from collections import OrderedDict
from loguru import logger

from config import config
from core.device_utils import get_device

# 延迟导入，避免启动时加载模型
sv_pipeline = None
_pipeline_lock = threading.Lock()


def get_sv_pipeline():
    """延迟加载声纹验证 pipeline，支持 GPU/CPU 自动检测并优先复用全局单例"""
    global sv_pipeline
    if sv_pipeline is None:
        with _pipeline_lock:
            # 双重检查锁定，防止多线程重复加载
            if sv_pipeline is None:
                # 1. 优先复用 engine_loader 中已预加载的 sv_pipeline
                try:
                    from models.engine_loader import engine_loader
                    if engine_loader.sv_pipeline is not None:
                        sv_pipeline = engine_loader.sv_pipeline
                        logger.info("[本地声纹] 成功复用 engine_loader 全局声纹模型")
                        return sv_pipeline
                except Exception:
                    pass

                from modelscope.pipelines import pipeline

                device = get_device()
                model_name = getattr(config, "speaker_sv_model", "iic/speech_eres2netv2_sv_zh-cn_16k-common")
                revision = getattr(config, "speaker_sv_revision", "v1.0.2")

                logger.info(f"[本地声纹] 正在加载声纹模型: {model_name} (设备: {device})...")
                sv_pipeline = pipeline(
                    task="speaker-verification",
                    model=model_name,
                    model_revision=revision,
                    device=device,
                )
                logger.info(f"[本地声纹] 模型加载完成 (设备: {device})")
    return sv_pipeline


class LocalVoiceprintClient:
    """
    本地声纹识别客户端
    基于 ModelScope ERes2Net 模型，支持注册、验证、删除等功能
    支持本地文件存储和 Milvus 向量数据库两种存储后端
    """

    def __init__(
        self,
        storage_dir: str = "voiceprint_db",
        threshold: float = 0.4,
        max_cache_size: int = 1000,
        preload_model: bool = True,
        # Milvus 配置参数
        milvus_enabled: bool = False,
        milvus_host: str = "localhost",
        milvus_port: int = 19530,
        milvus_database: str = "default",
        milvus_user: str = "",
        milvus_password: str = "",
        milvus_collection_prefix: str = "voiceprint",
    ):
        """
        初始化本地声纹客户端

        Args:
            storage_dir: 声纹数据存储目录（本地模式）
            threshold: 声纹匹配阈值，默认 0.4
            max_cache_size: 最大缓存数量，默认 1000
            preload_model: 是否在初始化时预加载模型，默认 True
            milvus_enabled: 是否启用 Milvus 存储后端
            milvus_host: Milvus 服务地址
            milvus_port: Milvus 服务端口
            milvus_database: Milvus 数据库名称
            milvus_user: Milvus 用户名
            milvus_password: Milvus 密码
            milvus_collection_prefix: Milvus Collection 名称前缀
        """
        self.storage_dir = storage_dir
        self.threshold = threshold
        self.max_cache_size = max_cache_size
        self.features_dir = os.path.join(storage_dir, "features")
        self.audio_dir = os.path.join(storage_dir, "audio")
        self.db_file = os.path.join(storage_dir, "voiceprint_db.json")

        # Milvus 相关
        self.milvus_enabled = milvus_enabled
        self._milvus_client = None

        # 尝试初始化 Milvus 客户端
        if milvus_enabled:
            try:
                from modules.milvus_voiceprint import MilvusVoiceprintClient

                logger.info(f"[本地声纹] ✅ MILVUS_ENABLED=true，正在连接 Milvus...")
                logger.info(f"[本地声纹] 连接参数: host={milvus_host}, port={milvus_port}, database={milvus_database}")
                
                self._milvus_client = MilvusVoiceprintClient(
                    host=milvus_host,
                    port=milvus_port,
                    database=milvus_database,
                    user=milvus_user,
                    password=milvus_password,
                    collection_prefix=milvus_collection_prefix,
                    threshold=threshold,
                    preload_model=preload_model,
                )
                logger.info(f"[本地声纹] ✅ Milvus 模式已启用，数据库: {milvus_database}")
            except Exception as e:
                logger.warning(f"[本地声纹] ⚠️ Milvus 初始化失败: {e}")
                logger.warning(f"[本地声纹] ⚠️ 降级到本地文件存储模式: {storage_dir}")
                self.milvus_enabled = False
                self._milvus_client = None
        else:
            logger.info(f"[本地声纹] 📁 MILVUS_ENABLED=false，使用本地文件存储模式: {storage_dir}")

        # 创建存储目录（本地模式需要）
        os.makedirs(self.features_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)

        # 加载数据库
        self.db = self._load_db()

        # 内存中的 embedding 缓存，使用 OrderedDict 实现 LRU
        self.embeddings_cache: OrderedDict[str, np.ndarray] = OrderedDict()

        # 预加载模型（避免第一次请求时加载）- 仅在本地模式或 Milvus 未预加载时
        if preload_model and not self.milvus_enabled:
            logger.info("[本地声纹] 预加载模型...")
            get_sv_pipeline()

        mode = "Milvus" if self.milvus_enabled else "本地文件"
        logger.info(f"[本地声纹] 客户端初始化完成，模式: {mode}, 存储目录: {storage_dir}, 最大缓存: {max_cache_size}")

    def _load_db(self) -> dict:
        """加载声纹数据库"""
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"[本地声纹] 加载数据库失败: {e}")
        return {"groups": {}}

    def _save_db(self):
        """保存声纹数据库"""
        try:
            with open(self.db_file, "w", encoding="utf-8") as f:
                json.dump(self.db, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"[本地声纹] 保存数据库失败: {e}")

    def _convert_audio_to_bytes(self, audio_data: Union[bytes, np.ndarray]) -> bytes:
        """
        统一转换音频为 bytes（16k16bit PCM）

        Args:
            audio_data: 音频数据（bytes 或 numpy 数组）

        Returns:
            PCM bytes
        """
        if isinstance(audio_data, np.ndarray):
            if audio_data.dtype in [np.float32, np.float64]:
                audio_data = np.clip(audio_data, -1.0, 1.0)
                audio_data = (audio_data * 32767).astype(np.int16)
            else:
                audio_data = audio_data.astype(np.int16)
            return audio_data.tobytes()
        return audio_data

    def _audio_to_wav_file(self, audio_data: Union[bytes, np.ndarray]) -> str:
        """
        将音频数据转换为临时 WAV 文件

        Args:
            audio_data: PCM 音频数据（bytes 或 numpy 数组）

        Returns:
            临时 WAV 文件路径
        """
        # 转换为 bytes
        audio_bytes = self._convert_audio_to_bytes(audio_data)

        # 创建临时 WAV 文件
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            with wave.open(temp_file.name, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_bytes)
            return temp_file.name
        except Exception as e:
            os.unlink(temp_file.name)
            raise e

    def _extract_embedding(
        self, audio_data: Union[bytes, np.ndarray, str]
    ) -> Optional[np.ndarray]:
        """
        提取音频的声纹 embedding (内存直推 + 单次前向推理 + 自动设备对齐 + 长度截断)

        Args:
            audio_data: 音频数据（PCM bytes、numpy 数组或文件路径）

        Returns:
            声纹 embedding 向量，失败返回 None
        """
        temp_file = None
        try:
            pipeline = get_sv_pipeline()
            if pipeline is None:
                logger.error("[本地声纹] 声纹模型未就绪")
                return None

            # 1. 尝试将输入直接转换为 16kHz float32 numpy 数组 [-1.0, 1.0]
            audio_arr = None
            if isinstance(audio_data, str) and os.path.exists(audio_data):
                try:
                    import soundfile as sf
                    wav, sr = sf.read(audio_data)
                    if len(wav.shape) > 1:
                        wav = np.mean(wav, axis=1)
                    if sr != 16000:
                        import torchaudio
                        t_wav, _ = torchaudio.load(audio_data)
                        if t_wav.shape[0] > 1:
                            t_wav = torch.mean(t_wav, dim=0, keepdim=True)
                        t_wav = torchaudio.transforms.Resample(sr, 16000)(t_wav)
                        audio_arr = t_wav[0].numpy()
                    else:
                        audio_arr = wav.astype(np.float32)
                except Exception:
                    audio_arr = None
            elif isinstance(audio_data, bytes):
                int16_arr = np.frombuffer(audio_data, dtype=np.int16)
                audio_arr = int16_arr.astype(np.float32) / 32767.0
            elif isinstance(audio_data, np.ndarray):
                audio_arr = audio_data.astype(np.float32)
                if audio_arr.dtype == np.int16 or (len(audio_arr) > 0 and np.max(np.abs(audio_arr)) > 1.0):
                    audio_arr = audio_arr / 32767.0

            # 2. 内存直通极速推理（单次 forward，Kaldi fbank + CNN 在内存直接完成）
            if audio_arr is not None and len(audio_arr) > 0:
                # 截断有效语音（最长 6 秒 = 96000 个采样点，削减长音频计算量，避免无谓消耗）
                max_samples = 16000 * 6
                if len(audio_arr) > max_samples:
                    audio_arr = audio_arr[:max_samples]

                model = getattr(pipeline, "model", None)
                if model is not None and callable(model):
                    with torch.no_grad():
                        emb_res = model(audio_arr)
                    if isinstance(emb_res, torch.Tensor):
                        return emb_res.squeeze().detach().cpu().numpy()
                    elif isinstance(emb_res, np.ndarray):
                        return emb_res.squeeze()
                    elif isinstance(emb_res, list) and len(emb_res) > 0:
                        item = emb_res[0]
                        return item.squeeze().detach().cpu().numpy() if isinstance(item, torch.Tensor) else np.array(item).squeeze()

            # 3. 兜底回退：如果直通失败，走原有 pipeline 文件途径
            logger.debug("[本地声纹] 直通推理未生效，降级使用 pipeline 提取")
            if isinstance(audio_data, str) and os.path.exists(audio_data):
                audio_path = audio_data
            else:
                temp_file = self._audio_to_wav_file(audio_data)
                audio_path = temp_file

            result = pipeline([audio_path, audio_path], output_emb=True)
            if "embs" in result and len(result["embs"]) > 0:
                return np.array(result["embs"][0])

            logger.warning("[本地声纹] 提取 embedding 失败")
            return None

        except Exception as e:
            logger.error(f"[本地声纹] 提取 embedding 异常: {e}")
            return None
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass

    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        计算两个 embedding 的余弦相似度

        Args:
            emb1: 第一个 embedding
            emb2: 第二个 embedding

        Returns:
            相似度分数
        """
        emb1 = emb1.flatten()
        emb2 = emb2.flatten()
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0

        # 计算余弦相似度
        dot_product = np.dot(emb1, emb2)
        similarity = float(dot_product / (norm1 * norm2))

        # 调试信息：检查 embedding 范数和相似度范围
        logger.debug(
            f"[本地声纹] 相似度计算: norm1={norm1:.4f}, norm2={norm2:.4f}, "
            f"dot={dot_product:.4f}, similarity={similarity:.4f}"
        )

        return similarity

    def _get_embedding(self, group_id: str, feature_id: str) -> Optional[np.ndarray]:
        """获取缓存或加载 embedding，使用LRU策略"""
        cache_key = f"{group_id}_{feature_id}"

        if cache_key in self.embeddings_cache:
            # 移动到末尾（最近使用）
            self.embeddings_cache.move_to_end(cache_key)
            return self.embeddings_cache[cache_key]

        # 从文件加载
        emb_file = os.path.join(self.features_dir, group_id, f"{feature_id}.npy")
        if os.path.exists(emb_file):
            emb = np.load(emb_file)
            # 检查缓存大小，超出则移除最旧的
            if len(self.embeddings_cache) >= self.max_cache_size:
                # 移除最旧的项（OrderedDict的第一个）
                oldest_key = next(iter(self.embeddings_cache))
                del self.embeddings_cache[oldest_key]
                logger.debug(f"[本地声纹] 缓存已满，移除最旧项: {oldest_key}")
            self.embeddings_cache[cache_key] = emb
            return emb

        return None

    def create_group(
        self, group_id: str, group_name: str = "", group_info: str = ""
    ) -> bool:
        """
        创建声纹特征库

        Args:
            group_id: 特征库ID
            group_name: 特征库名称
            group_info: 特征库信息

        Returns:
            是否成功
        """
        # Milvus 模式委托
        if self.milvus_enabled and self._milvus_client:
            return self._milvus_client.create_group(group_id, group_name, group_info)

        try:
            if group_id in self.db["groups"]:
                logger.warning(f"[本地声纹] 特征库已存在: {group_id}")
                return True

            self.db["groups"][group_id] = {
                "name": group_name or group_id,
                "info": group_info,
                "features": {},
                "created_at": datetime.now().isoformat(),
            }

            # 创建特征库目录
            os.makedirs(os.path.join(self.features_dir, group_id), exist_ok=True)
            os.makedirs(os.path.join(self.audio_dir, group_id), exist_ok=True)

            self._save_db()
            logger.info(f"[本地声纹] 特征库创建成功: {group_id}")
            return True

        except Exception as e:
            logger.error(f"[本地声纹] 创建特征库失败: {e}")
            return False

    def create_feature(
        self,
        group_id: str,
        feature_id: str,
        audio_data: Union[bytes, np.ndarray],
        feature_info: str = "",
    ) -> dict:
        """
        创建声纹特征（注册用户声纹）

        Args:
            group_id: 特征库ID
            feature_id: 特征ID（用户标识），如果为空则自动生成
            audio_data: 音频二进制数据
            feature_info: 特征信息（可存储用户名等）

        Returns:
            {"success": True, "featureId": "xxx"} 或 {"success": False, "error": "xxx"}
        """
        # Milvus 模式委托
        if self.milvus_enabled and self._milvus_client:
            return self._milvus_client.create_feature(group_id, feature_id, audio_data, feature_info)

        try:
            # 确保特征库存在
            if group_id not in self.db["groups"]:
                self.create_group(group_id)

            # 如果没有提供 feature_id，自动生成
            if not feature_id:
                feature_id = str(uuid.uuid4())[:8]

            # 提取 embedding
            embedding = self._extract_embedding(audio_data)
            if embedding is None:
                return {"success": False, "error": "提取声纹特征失败"}

            # 保存 embedding
            emb_file = os.path.join(self.features_dir, group_id, f"{feature_id}.npy")
            np.save(emb_file, embedding)

            # 保存音频文件（可选，用于调试）
            audio_file = os.path.join(self.audio_dir, group_id, f"{feature_id}.wav")
            audio_bytes = self._convert_audio_to_bytes(audio_data)

            with wave.open(audio_file, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_bytes)

            # 更新数据库
            self.db["groups"][group_id]["features"][feature_id] = {
                "info": feature_info or feature_id,
                "created_at": datetime.now().isoformat(),
            }
            self._save_db()

            # 更新缓存
            cache_key = f"{group_id}_{feature_id}"
            self.embeddings_cache[cache_key] = embedding

            logger.info(f"[本地声纹] 特征创建成功: {feature_id}")
            return {"success": True, "featureId": feature_id}

        except Exception as e:
            logger.error(f"[本地声纹] 创建特征失败: {e}")
            return {"success": False, "error": str(e)}

    def verify(
        self, group_id: str, feature_id: str, audio_data: Union[bytes, np.ndarray]
    ) -> dict:
        """
        1:1验证 - 与指定特征对比

        Args:
            group_id: 特征库ID
            feature_id: 目标特征ID
            audio_data: 音频二进制数据

        Returns:
            {"score": 0.xx, "text": "yes/no", "featureId": "xxx"} 或 {"error": "xxx"}
        """
        # Milvus 模式委托
        if self.milvus_enabled and self._milvus_client:
            return self._milvus_client.verify(group_id, feature_id, audio_data)

        try:
            # 获取已注册的 embedding
            registered_emb = self._get_embedding(group_id, feature_id)
            if registered_emb is None:
                return {"error": f"特征不存在: {feature_id}"}

            # 提取待验证音频的 embedding
            test_emb = self._extract_embedding(audio_data)
            if test_emb is None:
                return {"error": "提取声纹特征失败"}

            # 计算相似度
            score = self._compute_similarity(registered_emb, test_emb)
            is_match = score >= self.threshold

            logger.info(f"[本地声纹] 1:1验证: score={score:.4f}, match={is_match}")
            return {
                "score": round(score, 5),
                "text": "yes" if is_match else "no",
                "featureId": feature_id,
            }

        except Exception as e:
            logger.error(f"[本地声纹] 1:1验证失败: {e}")
            return {"error": str(e)}

    def search(
        self,
        group_id: str,
        audio_data: Union[bytes, np.ndarray],
        top_k: int = 1,
        candidate_feature_ids: Optional[List[Dict[str, str]]] = None,
    ) -> dict:
        """
        1:N识别 - 与库中特征对比

        Args:
            group_id: 特征库ID
            audio_data: 音频二进制数据
            top_k: 返回前K个最匹配的结果
            candidate_feature_ids: 候选特征ID列表 [{"feature_id": "xxx", ...}]，
                                   提供时仅比对候选列表中的特征，未提供时比对全部特征

        Returns:
            {"scoreList": [{"featureId": "xxx", "score": 0.xx}]} 或 {"error": "xxx"}
        """
        # Milvus 模式委托
        if self.milvus_enabled and self._milvus_client:
            res = self._milvus_client.search(group_id, audio_data, top_k, candidate_feature_ids)
            if res and "scoreList" in res:
                self.db = self._load_db()
                local_info_map = {}
                for g in self.db.get("groups", {}).values():
                    for fid, finfo in g.get("features", {}).items():
                        if isinstance(finfo, dict) and finfo.get("info"):
                            local_info_map[fid] = finfo["info"]
                for item in res["scoreList"]:
                    if not item.get("featureInfo") and item.get("featureId") in local_info_map:
                        item["featureInfo"] = local_info_map[item["featureId"]]
            return res

        try:
            # 重新加载数据库以获取最新数据（支持多进程/多服务实时更新）
            self.db = self._load_db()

            if group_id not in self.db["groups"]:
                return {"error": f"特征库不存在: {group_id}"}

            features = self.db["groups"][group_id].get("features", {})
            if not features:
                return {"scoreList": []}

            # 确定比对范围：如果提供了候选列表，则仅比对候选列表中的特征
            if candidate_feature_ids:
                candidate_fid_set = {
                    c["feature_id"]
                    for c in candidate_feature_ids
                    if c.get("feature_id")
                }
                target_fids = [fid for fid in features.keys() if fid in candidate_fid_set]
                logger.info(
                    f"[本地声纹] 1:N识别(候选过滤): 候选 {len(candidate_fid_set)} 个, "
                    f"库中匹配 {len(target_fids)} 个, 库总量 {len(features)} 个"
                )
                if not target_fids:
                    logger.warning("[本地声纹] 候选特征在库中均不存在，返回空结果")
                    return {"scoreList": []}
            else:
                target_fids = list(features.keys())

            # 提取待识别音频的 embedding
            test_emb = self._extract_embedding(audio_data)
            if test_emb is None:
                return {"error": "提取声纹特征失败"}

            # 与目标特征对比
            scores = []
            for fid in target_fids:
                registered_emb = self._get_embedding(group_id, fid)
                if registered_emb is not None:
                    score = self._compute_similarity(registered_emb, test_emb)
                    scores.append(
                        {
                            "featureId": fid,
                            "score": round(score, 5),
                            "featureInfo": features[fid].get("info", ""),
                        }
                    )

            # 按分数排序
            scores.sort(key=lambda x: x["score"], reverse=True)
            score_list = scores[:top_k]

            logger.info(f"[本地声纹] 1:N识别: 找到 {len(score_list)} 个匹配")
            return {"scoreList": score_list}

        except Exception as e:
            logger.error(f"[本地声纹] 1:N识别失败: {e}")
            return {"error": str(e)}

    def update_feature(
        self,
        group_id: str,
        feature_id: str,
        audio_data: Union[bytes, np.ndarray],
        feature_info: str = "",
    ) -> bool:
        """
        更新声纹特征

        Args:
            group_id: 特征库ID
            feature_id: 特征ID
            audio_data: 音频二进制数据
            feature_info: 特征信息

        Returns:
            是否成功
        """
        # Milvus 模式委托
        if self.milvus_enabled and self._milvus_client:
            return self._milvus_client.update_feature(group_id, feature_id, audio_data, feature_info)

        try:
            if group_id not in self.db["groups"]:
                logger.warning(f"[本地声纹] 特征库不存在: {group_id}")
                return False

            if feature_id not in self.db["groups"][group_id].get("features", {}):
                logger.warning(f"[本地声纹] 特征不存在: {feature_id}")
                return False

            # 提取新的 embedding
            embedding = self._extract_embedding(audio_data)
            if embedding is None:
                return False

            # 更新 embedding 文件
            emb_file = os.path.join(self.features_dir, group_id, f"{feature_id}.npy")
            np.save(emb_file, embedding)

            # 更新数据库
            if feature_info:
                self.db["groups"][group_id]["features"][feature_id][
                    "info"
                ] = feature_info
            self.db["groups"][group_id]["features"][feature_id][
                "updated_at"
            ] = datetime.now().isoformat()
            self._save_db()

            # 更新缓存
            cache_key = f"{group_id}_{feature_id}"
            self.embeddings_cache[cache_key] = embedding

            logger.info(f"[本地声纹] 特征更新成功: {feature_id}")
            return True

        except Exception as e:
            logger.error(f"[本地声纹] 更新特征失败: {e}")
            return False

    def delete_feature(self, group_id: str, feature_id: str) -> bool:
        """
        删除声纹特征

        Args:
            group_id: 特征库ID
            feature_id: 特征ID

        Returns:
            是否成功
        """
        # Milvus 模式委托
        if self.milvus_enabled and self._milvus_client:
            return self._milvus_client.delete_feature(group_id, feature_id)

        try:
            if group_id not in self.db["groups"]:
                logger.warning(f"[本地声纹] 特征库不存在: {group_id}")
                return False

            if feature_id not in self.db["groups"][group_id].get("features", {}):
                logger.warning(f"[本地声纹] 特征不存在: {feature_id}")
                return False

            # 删除 embedding 文件
            emb_file = os.path.join(self.features_dir, group_id, f"{feature_id}.npy")
            if os.path.exists(emb_file):
                os.remove(emb_file)

            # 删除音频文件
            audio_file = os.path.join(self.audio_dir, group_id, f"{feature_id}.wav")
            if os.path.exists(audio_file):
                os.remove(audio_file)

            # 更新数据库
            del self.db["groups"][group_id]["features"][feature_id]
            self._save_db()

            # 清除缓存
            cache_key = f"{group_id}_{feature_id}"
            self.embeddings_cache.pop(cache_key, None)

            logger.info(f"[本地声纹] 特征删除成功: {feature_id}")
            return True

        except Exception as e:
            logger.error(f"[本地声纹] 删除特征失败: {e}")
            return False

    def query_feature_list(self, group_id: str) -> list:
        """
        查询特征库中的所有特征

        Args:
            group_id: 特征库ID

        Returns:
            特征列表 [{"featureId": "xxx", "featureInfo": "xxx"}]
        """
        # Milvus 模式委托
        if self.milvus_enabled and self._milvus_client:
            features = self._milvus_client.query_feature_list(group_id)
            # 从本地元数据数据库自动补充特征备注 (featureInfo)
            local_info_map = {}
            for g in self.db.get("groups", {}).values():
                for fid, finfo in g.get("features", {}).items():
                    if isinstance(finfo, dict) and finfo.get("info"):
                        local_info_map[fid] = finfo["info"]
            for f in features:
                if not f.get("featureInfo") and f.get("featureId") in local_info_map:
                    f["featureInfo"] = local_info_map[f["featureId"]]
            return features

        try:
            if group_id not in self.db["groups"]:
                logger.warning(f"[本地声纹] 特征库不存在: {group_id}")
                return []

            features = self.db["groups"][group_id].get("features", {})
            result = [
                {"featureId": fid, "featureInfo": info.get("info", "")}
                for fid, info in features.items()
            ]

            logger.info(f"[本地声纹] 查询到 {len(result)} 个特征")
            return result

        except Exception as e:
            logger.error(f"[本地声纹] 查询特征列表失败: {e}")
            return []

    def delete_group(self, group_id: str) -> bool:
        """
        删除整个特征库

        Args:
            group_id: 特征库ID

        Returns:
            是否成功
        """
        # Milvus 模式委托
        if self.milvus_enabled and self._milvus_client:
            return self._milvus_client.delete_group(group_id)

        try:
            if group_id not in self.db["groups"]:
                logger.warning(f"[本地声纹] 特征库不存在: {group_id}")
                return False

            # 删除所有特征文件
            import shutil

            emb_dir = os.path.join(self.features_dir, group_id)
            if os.path.exists(emb_dir):
                shutil.rmtree(emb_dir)

            audio_dir = os.path.join(self.audio_dir, group_id)
            if os.path.exists(audio_dir):
                shutil.rmtree(audio_dir)

            # 清除缓存
            keys_to_remove = [
                k for k in self.embeddings_cache if k.startswith(f"{group_id}_")
            ]
            for k in keys_to_remove:
                del self.embeddings_cache[k]

            # 更新数据库
            del self.db["groups"][group_id]
            self._save_db()

            logger.info(f"[本地声纹] 特征库删除成功: {group_id}")
            return True

        except Exception as e:
            logger.error(f"[本地声纹] 删除特征库失败: {e}")
            return False

    def health_check(self) -> dict:
        """
        健康检查

        Returns:
            {"status": "ok", "backend": "milvus/local", ...} 或 {"status": "error", "error": "xxx"}
        """
        if self.milvus_enabled and self._milvus_client:
            return self._milvus_client.health_check()

        return {
            "status": "ok",
            "backend": "local",
            "storage_dir": self.storage_dir,
        }

    def get_stats(self, group_id: str) -> dict:
        """
        获取特征库统计信息

        Args:
            group_id: 特征库ID

        Returns:
            {"count": xxx, "groupId": "xxx"} 或 {"error": "xxx"}
        """
        if self.milvus_enabled and self._milvus_client:
            return self._milvus_client.get_stats(group_id)

        try:
            if group_id not in self.db["groups"]:
                return {"error": f"特征库不存在: {group_id}", "count": 0}

            features = self.db["groups"][group_id].get("features", {})
            return {
                "groupId": group_id,
                "count": len(features),
            }

        except Exception as e:
            logger.error(f"[本地声纹] 获取统计信息失败: {e}")
            return {"error": str(e), "count": 0}
