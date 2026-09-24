import time
import httpx
from typing import Optional, Dict
from loguru import logger
from config import config

# 内存缓存，避免对同一说话人频繁发起 HTTP 请求: { speaker_id: {"data": dict, "ts": float} }
_SPEAKER_CACHE: Dict[str, Dict] = {}
CACHE_TTL = 300  # 缓存 5 分钟


async def get_speaker_info_from_api(speaker_id: str) -> Optional[dict]:
    """
    通过第三方业务接口查询说话人详细档案信息（真实姓名、角色、头像等）
    带本地内存缓存，防止重复请求打垮业务系统。
    """
    if not speaker_id or speaker_id in ["未知", "未知用户"] or speaker_id.startswith("用户"):
        return None

    now = time.time()
    if speaker_id in _SPEAKER_CACHE:
        cached = _SPEAKER_CACHE[speaker_id]
        if now - cached["ts"] < CACHE_TTL:
            return cached["data"]

    base_url = getattr(config, "third_party_api_base_url", "http://localhost:8080").rstrip("/")
    api_url = f"{base_url}/ai/treatment/speaker/{speaker_id}"

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(api_url)
            if response.status_code == 200:
                data = response.json()
                if data.get("code") == 200 and data.get("data"):
                    raw_data = data["data"]

                    # 提取角色名称
                    role_name = ""
                    roles = raw_data.get("roles", [])
                    if roles and isinstance(roles, list) and len(roles) > 0:
                        role_name = roles[0].get("roleName", "")

                    result = {
                        "speaker_id": speaker_id,
                        "speaker_name": raw_data.get("realName") or raw_data.get("username") or speaker_id,
                        "role": role_name,
                        "avatar_url": raw_data.get("avatar", ""),
                        "dept_name": raw_data.get("primaryDept", {}).get("deptName", "") if isinstance(raw_data.get("primaryDept"), dict) else "",
                    }

                    _SPEAKER_CACHE[speaker_id] = {"data": result, "ts": now}
                    logger.info(f"[说话人信息] 获取成功: {speaker_id} -> {result['speaker_name']} ({result['role']})")
                    return result
            else:
                logger.warning(f"[说话人信息] 第三方接口返回 HTTP {response.status_code} URL={api_url}")

    except Exception as e:
        logger.warning(f"[说话人信息] 查询第三方接口异常 (speaker_id={speaker_id}): {e}")

    # 查询失败时构建保底数据，并加入短期缓存避免密集重试
    fallback = {
        "speaker_id": speaker_id,
        "speaker_name": speaker_id,
        "role": "",
        "avatar_url": "",
        "dept_name": "",
    }
    _SPEAKER_CACHE[speaker_id] = {"data": fallback, "ts": now}
    return fallback


# 候选人声纹缓存: { user_id: {"data": list, "ts": float} }
_CANDIDATE_CACHE: Dict[str, Dict] = {}


async def get_candidate_feature_ids(user_id: Optional[str]) -> list:
    """
    通过第三方业务接口获取指定用户的声纹ID列表（用于定向缩小声纹比对范围，大幅提高速度与准确率）
    接口: GET {third_party_api_base_url}/ai/treatment/speaker/voiceprints?userId={user_id}

    Returns:
        候选特征ID列表 [{"feature_id": "xxx", "user_id": "yyy"}, ...]
    """
    if not user_id:
        return []

    now = time.time()
    if user_id in _CANDIDATE_CACHE:
        cached = _CANDIDATE_CACHE[user_id]
        if now - cached["ts"] < CACHE_TTL:
            return cached["data"]

    base_url = getattr(config, "third_party_api_base_url", "http://localhost:8080").rstrip("/")
    api_url = f"{base_url}/ai/treatment/speaker/voiceprints?userId={user_id}"

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(api_url)
            if response.status_code == 200:
                data = response.json()
                if data.get("code") == 200 and data.get("data"):
                    voiceprints = data["data"].get("voiceprints", [])
                    result = [
                        {
                            "feature_id": vp.get("speakerId", ""),
                            "user_id": data["data"].get("userId", user_id),
                        }
                        for vp in voiceprints
                        if vp.get("speakerId")
                    ]
                    _CANDIDATE_CACHE[user_id] = {"data": result, "ts": now}
                    logger.info(
                        f"[候选声纹] 获取成功: user_id={user_id}, 包含 {len(result)} 个候选特征: {[r['feature_id'] for r in result]}"
                    )
                    return result
            else:
                logger.warning(f"[候选声纹] 第三方接口返回 HTTP {response.status_code} URL={api_url}")

    except Exception as e:
        logger.warning(f"[候选声纹] 查询第三方接口异常 (user_id={user_id}): {e}")

    # 出错时返回空列表并记录短期缓存（10秒）避免风暴
    _CANDIDATE_CACHE[user_id] = {"data": [], "ts": now - CACHE_TTL + 10}
    return []

