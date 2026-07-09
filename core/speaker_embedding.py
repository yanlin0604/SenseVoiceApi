import asyncio

import numpy as np
import torch

from models.engine_loader import engine_loader


async def extract_speaker_embedding(segment_audio: np.ndarray) -> np.ndarray:
    """提取说话人声纹 embedding。"""
    if not engine_loader.sv_pipeline or not hasattr(engine_loader.sv_pipeline, "model"):
        raise RuntimeError("声纹验证模型未加载")

    tensor_audio = torch.from_numpy(segment_audio.astype(np.float32)).unsqueeze(0)
    if hasattr(engine_loader.sv_pipeline.model, "device"):
        tensor_audio = tensor_audio.to(engine_loader.sv_pipeline.model.device)

    sv_res = await asyncio.to_thread(engine_loader.sv_pipeline.model, tensor_audio)
    emb = _read_embedding(sv_res)
    emb_flat = np.array(emb, dtype=np.float32).flatten()
    if np.linalg.norm(emb_flat) <= 0:
        raise RuntimeError("声纹 embedding 为空")
    return emb_flat


def _read_embedding(sv_res):
    if isinstance(sv_res, torch.Tensor):
        return sv_res.detach().cpu().numpy()
    if isinstance(sv_res, np.ndarray):
        return sv_res
    if isinstance(sv_res, list) and len(sv_res) > 0:
        first = sv_res[0]
        if isinstance(first, torch.Tensor):
            return first.detach().cpu().numpy()
        return first
    raise RuntimeError(f"无法读取声纹模型返回结构: {type(sv_res)}")
