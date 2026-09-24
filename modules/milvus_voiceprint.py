"""
Milvus 声纹识别客户端模块
基于 Milvus 向量数据库实现高性能声纹注册、验证、检索等功能
"""

import os
import uuid
import tempfile
import threading
import numpy as np
import wave
from datetime import datetime
from typing import Optional, Union, List, Dict, Any
import torch
from loguru import logger

from config import config
from core.device_utils import get_device

# Milvus 客户端延迟导入
milvus_available = False
try:
    from pymilvus import (
        connections,
        Collection,
        FieldSchema,
        CollectionSchema,
        DataType,
        utility,
    )
    milvus_available = True
except ImportError:
    logger.warning("[Milvus声纹] pymilvus 未安装，Milvus 功能不可用")

# 延迟导入声纹模型
sv_pipeline = None
_pipeline_lock = threading.Lock()


def get_sv_pipeline():
    """延迟加载声纹验证 pipeline，支持 GPU/CPU 自动检测并优先复用全局单例"""
    global sv_pipeline
    if sv_pipeline is None:
        with _pipeline_lock:
            if sv_pipeline is None:
                # 1. 优先复用 engine_loader 中已预加载的 sv_pipeline
                try:
                    from models.engine_loader import engine_loader
                    if engine_loader.sv_pipeline is not None:
                        sv_pipeline = engine_loader.sv_pipeline
                        logger.info("[Milvus声纹] 成功复用 engine_loader 全局声纹模型")
                        return sv_pipeline
                except Exception:
                    pass

                from modelscope.pipelines import pipeline

                device = get_device()
                model_name = getattr(config, "speaker_sv_model", "iic/speech_eres2netv2_sv_zh-cn_16k-common")
                revision = getattr(config, "speaker_sv_revision", "v1.0.2")

                logger.info(f"[Milvus声纹] 正在加载声纹模型: {model_name} (设备: {device})...")
                sv_pipeline = pipeline(
                    task="speaker-verification",
                    model=model_name,
                    model_revision=revision,
                    device=device,
                )
                logger.info(f"[Milvus声纹] 模型加载完成 (设备: {device})")
    return sv_pipeline


class MilvusVoiceprintClient:
    """
    Milvus 声纹识别客户端
    基于 Milvus 向量数据库实现高性能声纹存储和检索
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        database: str = "default",
        user: str = "",
        password: str = "",
        collection_prefix: str = "voiceprint",
        threshold: float = 0.4,
        embedding_dim: int = 192,
        preload_model: bool = True,
    ):
        """
        初始化 Milvus 声纹客户端

        Args:
            host: Milvus 服务地址
            port: Milvus 服务端口
            database: Milvus 数据库名称
            user: Milvus 用户名（可选）
            password: Milvus 密码（可选）
            collection_prefix: Collection 名称前缀
            threshold: 声纹匹配阈值，默认 0.4
            embedding_dim: embedding 向量维度，默认 192（ERes2Net）
            preload_model: 是否预加载模型
        """
        if not milvus_available:
            raise ImportError("pymilvus 未安装，请运行: pip install pymilvus")

        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.collection_prefix = collection_prefix
        self.threshold = threshold
        self.embedding_dim = embedding_dim

        # Collection 缓存
        self._collections: Dict[str, Collection] = {}

        # 连接 Milvus
        self._connect()

        # 预加载模型
        if preload_model:
            logger.info("[Milvus声纹] 预加载模型...")
            get_sv_pipeline()

        logger.info(
            f"[Milvus声纹] 客户端初始化完成: {host}:{port}/{database}, "
            f"threshold={threshold}, embedding_dim={embedding_dim}"
        )

    def _connect(self):
        """连接 Milvus 服务"""
        try:
            alias = f"voiceprint_{self.host}_{self.port}_{self.database}"

            # 如果指定了非 default 数据库，尝试自动创建数据库
            if self.database and self.database != "default":
                temp_alias = f"temp_conn_{self.host}_{self.port}"
                try:
                    connections.connect(
                        alias=temp_alias,
                        host=self.host,
                        port=self.port,
                        user=self.user,
                        password=self.password,
                        db_name="default",
                    )
                    from pymilvus import db
                    databases = db.list_database(using=temp_alias)
                    if self.database not in databases:
                        logger.info(f"[Milvus声纹] 数据库 {self.database} 不存在，正在创建...")
                        db.create_database(self.database, using=temp_alias)
                        logger.info(f"[Milvus声纹] 数据库 {self.database} 创建成功")
                except Exception as dbe:
                    logger.warning(f"[Milvus声纹] 自动创建数据库 {self.database} 失败(可能版本不支持或无权限): {dbe}")
                finally:
                    try:
                        connections.disconnect(temp_alias)
                    except:
                        pass

            connections.connect(
                alias=alias,
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                db_name=self.database,
            )
            self._alias = alias
            logger.info(f"[Milvus声纹] 连接成功: {self.host}:{self.port}/{self.database}")
        except Exception as e:
            logger.error(f"[Milvus声纹] 连接失败: {e}")
            raise

    def _get_collection_name(self, group_id: str) -> str:
        """获取 Collection 名称 (防双下划线拼接)"""
        prefix = (self.collection_prefix or "voiceprint").rstrip("_")
        if group_id.startswith(f"{prefix}_"):
            return group_id
        return f"{prefix}_{group_id}"

    def _get_collection(self, group_id: str) -> Optional[Collection]:
        """获取或创建 Collection"""
        collection_name = self._get_collection_name(group_id)

        # 检查缓存
        if collection_name in self._collections:
            return self._collections[collection_name]

        # 检查 Collection 是否存在（严格匹配，严禁越界查找其他集合）
        if utility.has_collection(collection_name, using=self._alias):
            collection = Collection(collection_name, using=self._alias)
            collection.load()
            self._collections[collection_name] = collection
            return collection

        return None

    def _create_collection(self, group_id: str) -> Collection:
        """创建 Collection"""
        collection_name = self._get_collection_name(group_id)

        # 定义 Schema
        fields = [
            FieldSchema(name="feature_id", dtype=DataType.VARCHAR, max_length=256, is_primary=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="created_at", dtype=DataType.INT64),
        ]
        schema = CollectionSchema(fields, description=f"声纹特征库: {group_id}")

        # 创建 Collection
        collection = Collection(collection_name, schema, using=self._alias)

        # 创建索引（IVF_FLAT + COSINE）
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        collection.load()

        # 缓存
        self._collections[collection_name] = collection

        logger.info(f"[Milvus声纹] Collection 创建成功: {collection_name}")
        return collection

    def _convert_audio_to_bytes(self, audio_data: Union[bytes, np.ndarray]) -> bytes:
        """统一转换音频为 bytes（16k16bit PCM）"""
        if isinstance(audio_data, np.ndarray):
            if audio_data.dtype in [np.float32, np.float64]:
                audio_data = np.clip(audio_data, -1.0, 1.0)
                audio_data = (audio_data * 32767).astype(np.int16)
            else:
                audio_data = audio_data.astype(np.int16)
            return audio_data.tobytes()
        return audio_data

    def _audio_to_wav_file(self, audio_data: Union[bytes, np.ndarray]) -> str:
        """将音频数据转换为临时 WAV 文件"""
        audio_bytes = self._convert_audio_to_bytes(audio_data)

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
        """提取音频的声纹 embedding (内存直推 + 单次前向推理 + 自动设备对齐 + 长度截断)"""
        temp_file = None
        try:
            pipeline = get_sv_pipeline()
            if pipeline is None:
                logger.error("[Milvus声纹] 声纹模型未就绪")
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
            logger.debug("[Milvus声纹] 直通推理未生效，降级使用 pipeline 提取")
            if isinstance(audio_data, str) and os.path.exists(audio_data):
                audio_path = audio_data
            else:
                temp_file = self._audio_to_wav_file(audio_data)
                audio_path = temp_file

            result = pipeline([audio_path, audio_path], output_emb=True)
            if "embs" in result and len(result["embs"]) > 0:
                return np.array(result["embs"][0])

            logger.warning("[Milvus声纹] 提取 embedding 失败")
            return None

        except Exception as e:
            logger.error(f"[Milvus声纹] 提取 embedding 异常: {e}")
            return None
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass

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
        try:
            collection_name = self._get_collection_name(group_id)

            if utility.has_collection(collection_name, using=self._alias):
                logger.info(f"[Milvus声纹] 特征库已存在: {group_id}")
                return True

            self._create_collection(group_id)
            logger.info(f"[Milvus声纹] 特征库创建成功: {group_id}")
            return True

        except Exception as e:
            logger.error(f"[Milvus声纹] 创建特征库失败: {e}")
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
            feature_id: 特征ID（用户标识）
            audio_data: 音频二进制数据
            feature_info: 特征信息

        Returns:
            {"success": True, "featureId": "xxx"} 或 {"success": False, "error": "xxx"}
        """
        try:
            # 确保 Collection 存在
            collection = self._get_collection(group_id)
            if collection is None:
                collection = self._create_collection(group_id)

            # 如果没有提供 feature_id，自动生成
            if not feature_id:
                feature_id = str(uuid.uuid4())[:8]

            # 提取 embedding
            embedding = self._extract_embedding(audio_data)
            if embedding is None:
                return {"success": False, "error": "提取声纹特征失败"}

            # 插入数据
            data = [
                [feature_id],
                [embedding.tolist()],
                [int(datetime.now().timestamp())],
            ]
            collection.insert(data)
            collection.flush()

            logger.info(f"[Milvus声纹] 特征创建成功: {feature_id}")
            return {"success": True, "featureId": feature_id}

        except Exception as e:
            logger.error(f"[Milvus声纹] 创建特征失败: {e}")
            return {"success": False, "error": str(e)}

    def verify(
        self, group_id: str, feature_id: str, audio_data: Union[bytes, np.ndarray]
    ) -> dict:
        """
        1:1验证 - 与指定特征对比

        优化：使用 Milvus 向量搜索 + expr 过滤，让服务端计算相似度，
        避免传输已注册的 embedding，提升性能。

        Args:
            group_id: 特征库ID
            feature_id: 目标特征ID
            audio_data: 音频二进制数据

        Returns:
            {"score": 0.xx, "text": "yes/no", "featureId": "xxx"} 或 {"error": "xxx"}
        """
        try:
            collection = self._get_collection(group_id)
            if collection is None:
                return {"error": f"特征库不存在: {group_id}"}

            # 提取待验证音频的 embedding
            test_emb = self._extract_embedding(audio_data)
            if test_emb is None:
                return {"error": "提取声纹特征失败"}

            # 使用 Milvus 向量搜索 + 标量过滤
            # 让 Milvus 服务端计算相似度，避免传输已注册的 embedding
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
            results = collection.search(
                data=[test_emb.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=1,
                expr=f'feature_id == "{feature_id}"',
                output_fields=["feature_id"],
            )

            # 解析搜索结果
            if not results or len(results) == 0 or len(results[0]) == 0:
                return {"error": f"特征不存在: {feature_id}"}

            # Milvus 返回的 distance 就是 COSINE 相似度
            score = float(results[0][0].distance)
            matched_feature_id = results[0][0].entity.get("feature_id", feature_id)
            is_match = score >= self.threshold

            logger.info(f"[Milvus声纹] 1:1验证(向量搜索): score={score:.4f}, match={is_match}")
            return {
                "score": round(score, 5),
                "text": "yes" if is_match else "no",
                "featureId": matched_feature_id,
            }

        except Exception as e:
            logger.error(f"[Milvus声纹] 1:1验证失败: {e}")
            return {"error": str(e)}

    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """计算两个 embedding 的余弦相似度"""
        emb1 = emb1.flatten()
        emb2 = emb2.flatten()
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0

        dot_product = np.dot(emb1, emb2)
        similarity = float(dot_product / (norm1 * norm2))
        return similarity

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
            candidate_feature_ids: 候选特征ID列表

        Returns:
            {"scoreList": [{"featureId": "xxx", "score": 0.xx}]} 或 {"error": "xxx"}
        """
        try:
            collection = self._get_collection(group_id)
            if collection is None:
                return {"error": f"特征库不存在: {group_id}"}

            # 提取待识别音频的 embedding
            test_emb = self._extract_embedding(audio_data)
            if test_emb is None:
                return {"error": "提取声纹特征失败"}

            # 构建搜索参数
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}

            # 构建过滤表达式
            expr = None
            if candidate_feature_ids:
                candidate_fid_set = {
                    c["feature_id"]
                    for c in candidate_feature_ids
                    if c.get("feature_id")
                }
                if candidate_fid_set:
                    # 使用 in 操作符，更高效简洁
                    fid_list = list(candidate_fid_set)
                    expr = f'feature_id in {fid_list}'
                    logger.info(
                        f"[Milvus声纹] 1:N识别(候选过滤): 候选 {len(candidate_fid_set)} 个, 使用 in 操作符"
                    )

            # 执行搜索
            results = collection.search(
                data=[test_emb.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=["feature_id"],
            )

            # 解析结果
            score_list = []
            if results and len(results) > 0:
                for hit in results[0]:
                    score_list.append({
                        "featureId": hit.entity.get("feature_id"),
                        "score": round(hit.distance, 5),
                        "featureInfo": "",  # Milvus 不存储 feature_info
                    })

            logger.info(f"[Milvus声纹] 1:N识别: 找到 {len(score_list)} 个匹配")
            for item in score_list:
                logger.info(f"[Milvus声纹] 候选得分: {item['featureId']} -> {item['score']:.5f}")
            return {"scoreList": score_list}

        except Exception as e:
            logger.error(f"[Milvus声纹] 1:N识别失败: {e}")
            return {"error": str(e)}

    def update_feature(
        self,
        group_id: str,
        feature_id: str,
        audio_data: Union[bytes, np.ndarray],
        feature_info: str = "",
    ) -> bool:
        """
        更新声纹特征（删除旧的，插入新的）

        Args:
            group_id: 特征库ID
            feature_id: 特征ID
            audio_data: 音频二进制数据
            feature_info: 特征信息

        Returns:
            是否成功
        """
        try:
            collection = self._get_collection(group_id)
            if collection is None:
                logger.warning(f"[Milvus声纹] 特征库不存在: {group_id}")
                return False

            # 删除旧向量
            collection.delete(expr=f'feature_id == "{feature_id}"')

            # 提取新的 embedding
            embedding = self._extract_embedding(audio_data)
            if embedding is None:
                return False

            # 插入新向量
            data = [
                [feature_id],
                [embedding.tolist()],
                [int(datetime.now().timestamp())],
            ]
            collection.insert(data)
            collection.flush()

            logger.info(f"[Milvus声纹] 特征更新成功: {feature_id}")
            return True

        except Exception as e:
            logger.error(f"[Milvus声纹] 更新特征失败: {e}")
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
        try:
            collection = self._get_collection(group_id)
            if collection is None:
                logger.warning(f"[Milvus声纹] 特征库不存在: {group_id}")
                return False

            collection.delete(expr=f'feature_id == "{feature_id}"')
            collection.flush()

            logger.info(f"[Milvus声纹] 特征删除成功: {feature_id}")
            return True

        except Exception as e:
            logger.error(f"[Milvus声纹] 删除特征失败: {e}")
            return False

    def query_feature_list(self, group_id: str) -> list:
        """
        查询特征库中的所有特征

        Args:
            group_id: 特征库ID

        Returns:
            特征列表 [{"featureId": "xxx", "featureInfo": "xxx"}]
        """
        try:
            collection = self._get_collection(group_id)
            if collection is None:
                logger.info(f"[Milvus声纹] 特征库不存在: {group_id}")
                return []

            # 查询所有 feature_id
            results = collection.query(
                expr="created_at >= 0",
                output_fields=["feature_id", "created_at"],
                limit=1000,
            )

            result = [
                {"featureId": r["feature_id"], "featureInfo": "", "createdAt": r.get("created_at", 0)}
                for r in results
            ]

            logger.info(f"[Milvus声纹] 查询到 {len(result)} 个特征")
            return result

        except Exception as e:
            logger.error(f"[Milvus声纹] 查询特征列表失败: {e}")
            return []

    def delete_group(self, group_id: str) -> bool:
        """
        删除整个特征库

        Args:
            group_id: 特征库ID

        Returns:
            是否成功
        """
        try:
            collection_name = self._get_collection_name(group_id)

            if not utility.has_collection(collection_name, using=self._alias):
                logger.warning(f"[Milvus声纹] 特征库不存在: {group_id}")
                return False

            utility.drop_collection(collection_name, using=self._alias)

            # 清除缓存
            if collection_name in self._collections:
                del self._collections[collection_name]

            logger.info(f"[Milvus声纹] 特征库删除成功: {group_id}")
            return True

        except Exception as e:
            logger.error(f"[Milvus声纹] 删除特征库失败: {e}")
            return False

    def health_check(self) -> dict:
        """
        健康检查

        Returns:
            {"status": "ok", "backend": "milvus", "host": "xxx", "port": xxx} 或 {"status": "error", "error": "xxx"}
        """
        try:
            # 尝试获取版本信息
            version = utility.get_server_version(using=self._alias)
            return {
                "status": "ok",
                "backend": "milvus",
                "host": self.host,
                "port": self.port,
                "database": self.database,
                "version": version,
            }
        except Exception as e:
            return {
                "status": "error",
                "backend": "milvus",
                "error": str(e),
            }

    def get_stats(self, group_id: str) -> dict:
        """
        获取特征库统计信息

        Args:
            group_id: 特征库ID

        Returns:
            {"count": xxx, "groupId": "xxx"} 或 {"error": "xxx"}
        """
        try:
            collection = self._get_collection(group_id)
            if collection is None:
                return {"error": f"特征库不存在: {group_id}", "count": 0}

            stats = collection.num_entities
            return {
                "groupId": group_id,
                "count": stats,
            }

        except Exception as e:
            logger.error(f"[Milvus声纹] 获取统计信息失败: {e}")
            return {"error": str(e), "count": 0}
