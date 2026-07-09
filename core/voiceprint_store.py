import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
from loguru import logger

from config import config


@dataclass
class VoiceprintProfile:
    profile_id: str
    doctor_code: str
    speaker_name: str
    speaker_title: str = ""
    hospital_code: str = "default"
    dept_code: str = ""
    match_threshold: float = config.doctor_voiceprint_match_threshold
    is_active: bool = True

    @property
    def display_label(self) -> str:
        if self.speaker_title:
            return f"{self.speaker_title} {self.speaker_name}".strip()
        return self.speaker_name


@dataclass
class VoiceprintMatch:
    profile: VoiceprintProfile
    score: float


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    norm_left = np.linalg.norm(left)
    norm_right = np.linalg.norm(right)
    if norm_left <= 0 or norm_right <= 0:
        return -1.0
    return float(np.dot(left, right) / (norm_left * norm_right))


class LocalVoiceprintStore:
    """开发环境本地声纹库，生产环境应使用 Milvus。"""

    def __init__(self, store_path: str):
        self.store_path = store_path
        self.records = self._load()

    def upsert(self, profile: VoiceprintProfile, embedding: np.ndarray) -> None:
        self.records[profile.profile_id] = {
            "profile": asdict(profile),
            "embedding": embedding.astype(float).tolist(),
        }
        self._save()

    def search(
        self,
        embedding: np.ndarray,
        hospital_code: Optional[str] = None,
        dept_code: Optional[str] = None,
        limit: int = 5,
    ) -> Optional[VoiceprintMatch]:
        matches: list[VoiceprintMatch] = []
        for record in self.records.values():
            profile = VoiceprintProfile(**record["profile"])
            if not profile.is_active:
                continue
            if hospital_code and profile.hospital_code and profile.hospital_code != hospital_code:
                continue
            if dept_code and profile.dept_code and profile.dept_code != dept_code:
                continue
            score = cosine_similarity(embedding, np.array(record["embedding"], dtype=np.float32))
            if score >= profile.match_threshold:
                matches.append(VoiceprintMatch(profile=profile, score=score))
        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[0] if matches else None

    def match_profile(self, profile_id: str, embedding: np.ndarray) -> Optional[VoiceprintMatch]:
        record = self.records.get(profile_id)
        if not record:
            return None
        profile = VoiceprintProfile(**record["profile"])
        if not profile.is_active:
            return None
        score = cosine_similarity(embedding, np.array(record["embedding"], dtype=np.float32))
        if score >= profile.match_threshold:
            return VoiceprintMatch(profile=profile, score=score)
        return None

    def _load(self) -> dict:
        if not os.path.exists(self.store_path):
            return {}
        try:
            with open(self.store_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception as exc:
            logger.warning(f"读取本地声纹库失败，将使用空库: {exc}")
            return {}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.store_path)), exist_ok=True)
        with open(self.store_path, "w", encoding="utf-8") as file:
            json.dump(self.records, file, ensure_ascii=False)


class MilvusVoiceprintStore:
    def __init__(self):
        from pymilvus import connections

        connections.connect(
            alias=config.voiceprint_milvus_alias,
            host=config.voiceprint_milvus_host,
            port=str(config.voiceprint_milvus_port),
            db_name=config.voiceprint_milvus_database,
        )
        self.collection = None

    def upsert(self, profile: VoiceprintProfile, embedding: np.ndarray) -> None:
        collection = self._ensure_collection(len(embedding))
        collection.upsert([
            [profile.profile_id],
            [embedding.astype(float).tolist()],
            [profile.doctor_code or ""],
            [profile.speaker_name or ""],
            [profile.speaker_title or ""],
            [profile.hospital_code or ""],
            [profile.dept_code or ""],
            [float(profile.match_threshold)],
            [bool(profile.is_active)],
        ])
        collection.flush()

    def search(
        self,
        embedding: np.ndarray,
        hospital_code: Optional[str] = None,
        dept_code: Optional[str] = None,
        limit: int = 5,
    ) -> Optional[VoiceprintMatch]:
        collection = self._ensure_collection(len(embedding))
        expr = self._build_filter(hospital_code, dept_code)
        results = collection.search(
            data=[embedding.astype(float).tolist()],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {}},
            limit=limit,
            expr=expr,
            output_fields=[
                "profile_id", "doctor_code", "speaker_name", "speaker_title",
                "hospital_code", "dept_code", "match_threshold", "is_active",
            ],
        )
        if not results:
            return None
        for hit in results[0]:
            profile = self._profile_from_hit(hit)
            score = float(getattr(hit, "score", getattr(hit, "distance", -1.0)))
            if profile.is_active and score >= profile.match_threshold:
                return VoiceprintMatch(profile=profile, score=score)
        return None

    def match_profile(self, profile_id: str, embedding: np.ndarray) -> Optional[VoiceprintMatch]:
        collection = self._ensure_collection(len(embedding))
        expr = f'profile_id == "{self._escape(profile_id)}"'
        rows = collection.query(
            expr=expr,
            output_fields=[
                "profile_id", "embedding", "doctor_code", "speaker_name", "speaker_title",
                "hospital_code", "dept_code", "match_threshold", "is_active",
            ],
            limit=1,
        )
        if not rows:
            return None
        row = rows[0]
        profile = self._profile_from_row(row)
        if not profile.is_active:
            return None
        score = cosine_similarity(embedding, np.array(row["embedding"], dtype=np.float32))
        if score >= profile.match_threshold:
            return VoiceprintMatch(profile=profile, score=score)
        return None

    def _ensure_collection(self, dim: int):
        from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility

        if self.collection is not None:
            return self.collection

        name = config.voiceprint_milvus_collection
        if not utility.has_collection(name, using=config.voiceprint_milvus_alias):
            fields = [
                FieldSchema(name="profile_id", dtype=DataType.VARCHAR, is_primary=True, max_length=200),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema(name="doctor_code", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="speaker_name", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="speaker_title", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="hospital_code", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="dept_code", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="match_threshold", dtype=DataType.FLOAT),
                FieldSchema(name="is_active", dtype=DataType.BOOL),
            ]
            schema = CollectionSchema(fields=fields, description="MedAI 医生声纹库")
            collection = Collection(name=name, schema=schema, using=config.voiceprint_milvus_alias)
            collection.create_index("embedding", {"index_type": "AUTOINDEX", "metric_type": "COSINE", "params": {}})
        else:
            collection = Collection(name=name, using=config.voiceprint_milvus_alias)
        collection.load()
        self.collection = collection
        return collection

    def _build_filter(self, hospital_code: Optional[str], dept_code: Optional[str]) -> Optional[str]:
        filters = ["is_active == true"]
        if hospital_code:
            filters.append(f'hospital_code == "{self._escape(hospital_code)}"')
        if dept_code:
            escaped_dept = self._escape(dept_code)
            filters.append(f'(dept_code == "" or dept_code == "{escaped_dept}")')
        return " and ".join(filters)

    def _profile_from_hit(self, hit) -> VoiceprintProfile:
        entity = hit.entity
        return VoiceprintProfile(
            profile_id=str(entity.get("profile_id")),
            doctor_code=str(entity.get("doctor_code") or ""),
            speaker_name=str(entity.get("speaker_name") or ""),
            speaker_title=str(entity.get("speaker_title") or ""),
            hospital_code=str(entity.get("hospital_code") or "default"),
            dept_code=str(entity.get("dept_code") or ""),
            match_threshold=float(entity.get("match_threshold") or config.doctor_voiceprint_match_threshold),
            is_active=bool(entity.get("is_active")),
        )

    def _profile_from_row(self, row: dict) -> VoiceprintProfile:
        return VoiceprintProfile(
            profile_id=str(row.get("profile_id")),
            doctor_code=str(row.get("doctor_code") or ""),
            speaker_name=str(row.get("speaker_name") or ""),
            speaker_title=str(row.get("speaker_title") or ""),
            hospital_code=str(row.get("hospital_code") or "default"),
            dept_code=str(row.get("dept_code") or ""),
            match_threshold=float(row.get("match_threshold") or config.doctor_voiceprint_match_threshold),
            is_active=bool(row.get("is_active")),
        )

    def _escape(self, value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')


_voiceprint_store = None


def get_voiceprint_store():
    global _voiceprint_store
    if _voiceprint_store is not None:
        return _voiceprint_store

    if config.voiceprint_store_backend.lower() == "milvus":
        try:
            _voiceprint_store = MilvusVoiceprintStore()
            logger.info("医生声纹库使用 Milvus")
            return _voiceprint_store
        except Exception as exc:
            logger.warning(f"初始化 Milvus 声纹库失败，降级为本地声纹库: {exc}")

    _voiceprint_store = LocalVoiceprintStore(config.voiceprint_local_store_path)
    logger.info(f"医生声纹库使用本地 JSON: {config.voiceprint_local_store_path}")
    return _voiceprint_store
