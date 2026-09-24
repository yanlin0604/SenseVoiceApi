import os
import torch
from loguru import logger
from config import config

_current_device = None


def get_device(preferred: str = None) -> str:
    """
    自动检测可用计算设备：
    - auto (默认): 优先检测 GPU，若可用且可正常分配张量则使用 GPU，否则自动安全平滑降级为 CPU
    - cpu: 强制使用 CPU 运行
    - cuda / cuda:0: 强制指定 GPU 运行
    """
    global _current_device
    if _current_device is not None and preferred is None:
        return _current_device

    pref = (preferred or getattr(config, "device", "auto") or "auto").lower().strip()

    # 1. 显式指定使用 CPU
    if pref == "cpu":
        logger.info("[设备检测] 配置强制使用 CPU 运行")
        _current_device = "cpu"
        _tune_cpu_threads()
        return "cpu"

    # 2. 探测 GPU 是否真正可用
    if torch.cuda.is_available():
        target_device = "cuda:0" if pref == "auto" else pref
        try:
            # 探测性分配微小张量，验证驱动及显存上下文是否就绪（防止假可用或驱动崩溃）
            probe = torch.zeros(1, device=target_device)
            del probe
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"[设备检测] ✅ 成功检测到可用 GPU: {device_name} ({target_device})")
            _current_device = target_device
            return target_device
        except Exception as e:
            logger.warning(f"[设备检测] ⚠️ GPU 存在但无法分配张量 ({e})，安全降级至 CPU")
            _current_device = "cpu"
            _tune_cpu_threads()
            return "cpu"

    # 3. 未检测到 GPU，使用 CPU
    logger.info("[设备检测] 💻 未检测到可用 GPU，采用 CPU 模式运行")
    _current_device = "cpu"
    _tune_cpu_threads()
    return "cpu"


def _tune_cpu_threads():
    """在 CPU 模式下优化 PyTorch 线程数，防止多核抢占打满系统导致事件卡死"""
    try:
        cpu_count = os.cpu_count() or 4
        # 限制线程数为合理数量（如 2~4），既保证单核并发又留出系统余量
        threads = min(4, max(1, cpu_count // 2))
        torch.set_num_threads(threads)
        logger.debug(f"[设备检测] PyTorch CPU 线程数已设置为: {threads}")
    except Exception as e:
        logger.debug(f"[设备检测] 设置 PyTorch CPU 线程数失败: {e}")
