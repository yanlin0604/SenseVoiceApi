import asyncio
from contextlib import asynccontextmanager
from config import config
from loguru import logger


class InferenceLimiter:
    """
    模型推理并发限制器
    通过 asyncio.Semaphore 限制并发调用深度学习模型的协程数，
    避免多客户端并发时打满 GPU 显存或引发 OOM。
    """
    _sem_vad: asyncio.Semaphore | None = None
    _sem_asr: asyncio.Semaphore | None = None
    _sem_sv: asyncio.Semaphore | None = None

    @classmethod
    def get_sem_vad(cls) -> asyncio.Semaphore:
        if cls._sem_vad is None:
            cls._sem_vad = asyncio.Semaphore(config.max_concurrent_vad)
            logger.debug(f"[并发限流] 初始化 VAD 信号量, 容量: {config.max_concurrent_vad}")
        return cls._sem_vad

    @classmethod
    def get_sem_asr(cls) -> asyncio.Semaphore:
        if cls._sem_asr is None:
            cls._sem_asr = asyncio.Semaphore(config.max_concurrent_asr)
            logger.debug(f"[并发限流] 初始化 ASR 信号量, 容量: {config.max_concurrent_asr}")
        return cls._sem_asr

    @classmethod
    def get_sem_sv(cls) -> asyncio.Semaphore:
        if cls._sem_sv is None:
            cls._sem_sv = asyncio.Semaphore(config.max_concurrent_sv)
            logger.debug(f"[并发限流] 初始化 SV 信号量, 容量: {config.max_concurrent_sv}")
        return cls._sem_sv


@asynccontextmanager
async def limit_vad():
    """VAD 端点检测并发保护"""
    async with InferenceLimiter.get_sem_vad():
        yield


@asynccontextmanager
async def limit_asr():
    """ASR 语音转写并发保护"""
    async with InferenceLimiter.get_sem_asr():
        yield


@asynccontextmanager
async def limit_sv():
    """声纹识别/分离并发保护"""
    async with InferenceLimiter.get_sem_sv():
        yield
