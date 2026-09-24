import io
import os
import sys
import shutil
import subprocess
import uuid
import base64
import asyncio
import torch
import soundfile as sf
import numpy as np
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from loguru import logger

# 确保项目根目录在 sys.path 中，以便导入 modules
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import config

router = APIRouter(tags=["声纹管理与验证"])

# ==================== 请求与响应模型 ====================

class VoiceprintCreateGroupRequest(BaseModel):
    """创建声纹特征库请求"""
    group_id: str = Field(..., description="特征库ID")
    group_name: Optional[str] = Field("", description="特征库名称")
    group_info: Optional[str] = Field("", description="特征库信息")


class VoiceprintDeleteRequest(BaseModel):
    """删除声纹特征请求"""
    feature_ids: List[str] = Field(..., description="待删除的声纹特征ID列表")
    group_id: Optional[str] = Field(None, description="特征库ID（可选）")


# ==================== 后端客户端管理器 (100% 纯本地化) ====================

_local_voiceprint_client = None


def get_voiceprint_backend():
    """
    获取本地高性能声纹识别客户端实例（单例模式，纯本地化运行）
    """
    global _local_voiceprint_client

    if _local_voiceprint_client is None:
        from modules.local_voiceprint import LocalVoiceprintClient

        storage_dir = getattr(config, "local_voiceprint_dir", "voiceprint_db")
        # 如果是相对路径，解析到项目根目录
        if not os.path.isabs(storage_dir):
            storage_dir = os.path.join(ROOT_DIR, storage_dir)

        threshold = float(getattr(config, "local_voiceprint_threshold", 0.4))
        logger.info(
            f"[声纹模块] 初始化本地声纹客户端: 目录={storage_dir}, 阈值={threshold}, Milvus={config.milvus_enabled}"
        )
        _local_voiceprint_client = LocalVoiceprintClient(
            storage_dir=storage_dir,
            threshold=threshold,
            milvus_enabled=config.milvus_enabled,
            milvus_host=config.milvus_host,
            milvus_port=config.milvus_port,
            milvus_database=config.milvus_database,
            milvus_user=config.milvus_user,
            milvus_password=config.milvus_password,
            milvus_collection_prefix=config.milvus_collection_prefix,
        )
    return "local", _local_voiceprint_client


# ==================== 音频转换工具函数 ====================

def _ensure_torchaudio_ffmpeg_backend() -> None:
    """
    确保 torchaudio 的 ffmpeg 解码后端可用（用于解码 WebM/Opus 等
    MediaRecorder 产出的格式）。torchaudio 2.x 通过 libav* 动态库工作，
    一般随 conda 的 Library/bin 分发；系统 PATH 里有 ffmpeg 时也一并尝试。"""
    import torchaudio
    if not hasattr(torchaudio.utils, "ffmpeg_utils"):
        return
    if torchaudio.utils.ffmpeg_utils.get_versions().get("libavcodec"):
        return  # 后端已可用
    candidate_dirs = []
    for ffmpeg in ("ffmpeg.exe", "ffmpeg"):
        found = shutil.which(ffmpeg)
        if found:
            candidate_dirs.append(os.path.dirname(os.path.abspath(found)))
    # conda 环境布局: <env>/lib/os.py -> <env>/Library/bin 含 libav*.dll
    conda_bin = os.path.join(os.path.dirname(os.path.dirname(os.__file__)), "Library", "bin")
    if os.path.isdir(conda_bin):
        candidate_dirs.append(conda_bin)
    for d in candidate_dirs:
        if d and d not in os.environ.get("PATH", ""):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            try:
                torchaudio.utils.ffmpeg_utils.get_versions()
                if torchaudio.utils.ffmpeg_utils.get_versions().get("libavcodec"):
                    logger.info(f"[音频转换] 已启用 torchaudio ffmpeg 后端 ({d})")
                    return
            except Exception:
                pass


def _looks_like_webm(audio_bytes: bytes) -> bool:
    """EBML/Matroska (WebM) 魔数: 0x1A45DFA3"""
    return len(audio_bytes) >= 4 and audio_bytes[:4] == b"\x1A\x45\xDF\xA3"


def convert_audio_to_pcm(audio_bytes: bytes, filename: str = "") -> bytes:
    """
    将音频数据转换为 16kHz, 16-bit, 单声道 PCM 原始字节
    优先使用 torchaudio 进行高速多相滤波重采样，支持 WAV, MP3, FLAC, OGG,
    WEBM/OPUS 等格式（webm 需要 ffmpeg 后端），
    若不可用则平滑降级到 soundfile。"""
    # 优先使用 torchaudio 进行极速流式多相滤波重采样
    try:
        import torchaudio
        if _looks_like_webm(audio_bytes):
            _ensure_torchaudio_ffmpeg_backend()
        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        waveform = torch.clamp(waveform, -1.0, 1.0)
        pcm_int16 = (waveform * 32767.0).to(torch.int16)
        return pcm_int16.numpy().tobytes()
    except Exception as te:
        logger.warning(f"[音频转换] torchaudio 解码失败 ({te})，降级使用 soundfile")

    try:
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        logger.debug(f"[音频转换] 原始输入: 采样率={sample_rate}Hz, shape={audio_data.shape}, dtype={audio_data.dtype}")

        # 转为单声道
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        # 重采样到 16000Hz
        if sample_rate != 16000:
            try:
                from scipy import signal
                num_samples = int(round(len(audio_data) * 16000 / sample_rate))
                audio_data = signal.resample(audio_data, num_samples)
            except Exception:
                old_idx = np.arange(len(audio_data), dtype=np.float64)
                new_len = int(round(len(audio_data) * 16000 / sample_rate))
                if new_len <= 1 or len(old_idx) <= 1:
                    new_len = max(new_len, 2)
                    old_idx = np.linspace(0, 1, num=len(audio_data), dtype=np.float64)
                new_idx = np.linspace(0, len(audio_data) - 1, num=new_len, dtype=np.float64)
                audio_data = np.interp(new_idx, old_idx, audio_data.astype(np.float64))

        # 转换为 int16 格式
        if audio_data.dtype != np.int16:
            if audio_data.dtype in [np.float32, np.float64]:
                audio_data = np.clip(audio_data, -1.0, 1.0)
                audio_data = (audio_data * 32767).astype(np.int16)
            else:
                audio_data = audio_data.astype(np.int16)

        return audio_data.tobytes()

    except Exception as e:
        logger.warning(f"[音频转换] soundfile 解码失败 ({e})，尝试 ffmpeg 兕底")

    # 最后兜底：soundfile 也不认识时，尝试调用系统 ffmpeg 命令行转码
    # （MediaRecorder 的 WebM/Opus 最常见于浏览器录音场景）
    if _looks_like_webm(audio_bytes):
        pcm_via_ffmpeg = _transcode_webm_to_pcm(audio_bytes)
        if pcm_via_ffmpeg:
            return pcm_via_ffmpeg
        # WebM 头绝不能被当作裸 PCM（会算出垃圾特征、必然识别失败）
        raise ValueError(
            "无法解码 WebM/Opus 音频：torchaudio/soundfile 均失败且系统 ffmpeg 不可用，"
            "请改用 WAV/MP3 上传或安装 ffmpeg"
        )

    logger.warning("[音频转换] 所有解码方式均失败，尝试直接按原始字节当作 PCM 返回")
    return audio_bytes


def _transcode_webm_to_pcm(audio_bytes: bytes) -> Optional[bytes]:
    """调用系统 ffmpeg 将 WebM/Opus 音频转码为 16kHz/16bit/mono PCM"""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error",
             "-i", "pipe:0", "-f", "s16le", "-acodec", "pcm_s16le",
             "-ac", "1", "-ar", "16000", "pipe:1"],
            input=audio_bytes, capture_output=True, timeout=30,
        )
        if proc.returncode != 0 or not proc.stdout:
            logger.warning(f"[音频转换] ffmpeg 转码失败: {proc.stderr.decode(errors='ignore')[:200]}")
            return None
        logger.info(f"[音频转换] ffmpeg 已将 WebM/Opus 转为 PCM ({len(proc.stdout)} bytes)")
        return proc.stdout
    except Exception as fe:
        logger.warning(f"[音频转换] ffmpeg 转码异常: {fe}")
        return None


# ==================== 声纹 API 路由端点 ====================

@router.post("/api/voiceprint/group/create")
async def create_voiceprint_group(request: VoiceprintCreateGroupRequest):
    """
    创建声纹特征库
    """
    mode, client = get_voiceprint_backend()
    try:
        success = client.create_group(
            group_id=request.group_id,
            group_name=request.group_name or request.group_id,
            group_info=request.group_info or "",
        )
        if success:
            return JSONResponse(
                content={
                    "success": True,
                    "message": f"特征库 {request.group_id} 创建成功",
                    "mode": mode,
                }
            )
        else:
            return JSONResponse(
                content={"success": False, "message": "特征库创建失败"},
                status_code=400,
            )
    except Exception as e:
        logger.error(f"创建特征库失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/voiceprint/feature/register")
async def register_voiceprint_feature(
    audio_file: UploadFile = File(..., description="音频文件（支持WAV、MP3等格式）"),
    feature_id: str = Form(..., description="特征ID（用户唯一标识）"),
    group_id: Optional[str] = Form(None, description="特征库ID（可选，不填则使用默认特征库）"),
    feature_info: Optional[str] = Form(None, description="特征信息（可选，可存储用户名等）"),
):
    """
    注册声纹特征（本地高性能 ERes2Net 模型提取特征向量并入库）
    上传音频文件，后台自动转换为 16k16bit 单声道 PCM 并录入本地/Milvus声纹特征库
    """
    mode, client = get_voiceprint_backend()
    try:
        audio_bytes = await audio_file.read()
        pcm_bytes = await asyncio.to_thread(convert_audio_to_pcm, audio_bytes, audio_file.filename)
        target_group_id = group_id or config.default_group_id

        result = await asyncio.to_thread(
            client.create_feature,
            group_id=target_group_id,
            feature_id=feature_id,
            audio_data=pcm_bytes,
            feature_info=feature_info or feature_id,
        )
        success = result.get("success", False) if isinstance(result, dict) else bool(result)
        if not success:
            err_msg = result.get("error", "声纹特征注册失败") if isinstance(result, dict) else "注册失败"
            return JSONResponse(content={"success": False, "message": err_msg}, status_code=400)
        return JSONResponse(
            content={
                "success": True,
                "feature_id": feature_id,
                "group_id": target_group_id,
                "message": "声纹特征注册成功",
                "mode": mode,
            }
        )
    except Exception as e:
        logger.error(f"注册声纹特征失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/voiceprint/feature/update")
async def update_voiceprint_feature(
    audio_file: UploadFile = File(..., description="音频文件（支持WAV、MP3等格式）"),
    feature_id: str = Form(..., description="特征ID（用户唯一标识）"),
    group_id: Optional[str] = Form(None, description="特征库ID（可选）"),
    feature_info: Optional[str] = Form(None, description="特征信息（可选）"),
):
    """
    更新声纹特征
    """
    mode, client = get_voiceprint_backend()
    try:
        audio_bytes = await audio_file.read()
        pcm_bytes = await asyncio.to_thread(convert_audio_to_pcm, audio_bytes, audio_file.filename)
        target_group_id = group_id or config.default_group_id

        result = await asyncio.to_thread(
            client.update_feature,
            group_id=target_group_id,
            feature_id=feature_id,
            audio_data=pcm_bytes,
            feature_info=feature_info or feature_id,
        )
        success = result.get("success", False) if isinstance(result, dict) else bool(result)
        if not success:
            err_msg = result.get("error", "更新失败") if isinstance(result, dict) else "更新失败"
            return JSONResponse(content={"success": False, "message": err_msg}, status_code=400)

        return JSONResponse(
            content={
                "success": True,
                "feature_id": feature_id,
                "group_id": target_group_id,
                "message": "声纹特征更新成功",
                "mode": mode,
            }
        )
    except Exception as e:
        logger.error(f"更新声纹特征失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/voiceprint/feature/delete")
async def delete_voiceprint_features(request: VoiceprintDeleteRequest):
    """
    批量删除声纹特征
    """
    mode, client = get_voiceprint_backend()
    try:
        target_group_id = request.group_id or config.default_group_id
        results = []

        for fid in request.feature_ids:
            ok = client.delete_feature(group_id=target_group_id, feature_id=fid)
            results.append({"feature_id": fid, "success": ok})

        return JSONResponse(
            content={
                "success": all(r["success"] for r in results),
                "results": results,
                "mode": mode,
            }
        )
    except Exception as e:
        logger.error(f"删除声纹特征失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/voiceprint/feature/list")
async def list_voiceprint_features(group_id: Optional[str] = None):
    """
    查询指定特征库中的所有声纹特征
    """
    mode, client = get_voiceprint_backend()
    try:
        target_group_id = group_id or config.default_group_id
        features = client.query_feature_list(group_id=target_group_id)

        return JSONResponse(
            content={
                "success": True,
                "group_id": target_group_id,
                "features": features,
                "count": len(features),
                "mode": mode,
            }
        )
    except Exception as e:
        logger.error(f"查询特征列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/speaker/verify")
async def verify_speaker(audio_file: UploadFile = File(...)):
    """
    声纹识别登录 / 1:1 验证接口
    上传音频自动识别说话人身份
    """
    mode, client = get_voiceprint_backend()
    try:
        audio_bytes = await audio_file.read()
        pcm_bytes = await asyncio.to_thread(convert_audio_to_pcm, audio_bytes, audio_file.filename)
        target_group_id = config.default_group_id

        result = await asyncio.to_thread(client.search, group_id=target_group_id, audio_data=pcm_bytes, top_k=1)
        if "error" in result:
            return JSONResponse(status_code=400, content={"code": -1, "msg": f"识别失败: {result['error']}"})

        score_list = result.get("scoreList", [])
        if not score_list:
            return JSONResponse(status_code=400, content={"code": -1, "msg": "未找到匹配的用户", "data": None})

        top_match = score_list[0]
        feature_id = top_match.get("featureId", "")
        score = float(top_match.get("score", 0.0))
        threshold = float(getattr(client, "threshold", config.local_voiceprint_threshold))

        if score >= threshold:
            logger.info(f"[声纹识别] 识别成功: {feature_id}, 得分: {score:.4f} >= {threshold}")
            return JSONResponse(
                status_code=200,
                content={
                    "code": 0,
                    "msg": "识别成功",
                    "data": {
                        "speaker_id": feature_id,
                        "score": score,
                        "threshold": threshold,
                        "mode": "local",
                    },
                },
            )
        else:
            logger.warning(f"[声纹识别] 相似度不足: {score:.4f} < {threshold}")
            return JSONResponse(
                status_code=400,
                content={
                    "code": -1,
                    "msg": "相似度不足，无法识别身份",
                    "data": {"score": score, "threshold": threshold},
                },
            )

    except Exception as e:
        logger.error(f"[verify_speaker] 识别异常: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/voiceprint/identify")
async def identify_voiceprint(
    audio_file: UploadFile = File(..., description="音频文件（支持WAV、MP3等格式）"),
    group_id: Optional[str] = Form(None, description="特征库ID（可选）"),
    top_k: int = Form(1, description="返回前K个最匹配的结果"),
):
    """
    1:N 声纹识别检索
    """
    mode, client = get_voiceprint_backend()
    try:
        audio_bytes = await audio_file.read()
        pcm_bytes = await asyncio.to_thread(convert_audio_to_pcm, audio_bytes, audio_file.filename)
        target_group_id = group_id or config.default_group_id

        result = await asyncio.to_thread(client.search, group_id=target_group_id, audio_data=pcm_bytes, top_k=top_k)
        if "error" in result:
            return JSONResponse(content={"success": False, "message": result["error"]}, status_code=400)

        score_list = result.get("scoreList", [])
        threshold = float(getattr(client, "threshold", config.local_voiceprint_threshold))
        for item in score_list:
            item["matched"] = float(item.get("score", 0)) >= threshold

        return JSONResponse(
            content={
                "success": True,
                "scoreList": score_list,
                "count": len(score_list),
                "message": f"识别完成，找到 {len(score_list)} 个候选结果",
                "mode": mode,
            }
        )
    except Exception as e:
        logger.error(f"1:N 声纹识别失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 兼容旧端点 ====================

@router.post("/api/voiceprint/register")
async def legacy_register_voiceprint(
    audio_file: UploadFile = File(..., description="音频文件"),
    audio_type: str = Form("raw", description="音频类型"),
    uid: Optional[str] = Form(None, description="用户唯一标识（可选）"),
):
    """兼容旧版 /api/voiceprint/register 接口"""
    feature_id = uid or str(uuid.uuid4())
    return await register_voiceprint_feature(
        audio_file=audio_file,
        feature_id=feature_id,
        group_id=None,
        feature_info=feature_id,
    )


@router.post("/api/voiceprint/update")
async def legacy_update_voiceprint(
    audio_file: UploadFile = File(..., description="音频文件"),
    feature_id: str = Form(..., description="要更新的声纹ID"),
    audio_type: str = Form("raw", description="音频类型"),
):
    """兼容旧版 /api/voiceprint/update 接口"""
    return await update_voiceprint_feature(
        audio_file=audio_file,
        feature_id=feature_id,
        group_id=None,
        feature_info=None,
    )


@router.post("/api/voiceprint/delete")
async def legacy_delete_voiceprint(request: VoiceprintDeleteRequest):
    """兼容旧版 /api/voiceprint/delete 接口"""
    return await delete_voiceprint_features(request)
