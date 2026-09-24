import os
import torch
from loguru import logger

# 必须先引入 config，以确保 MODELSCOPE_CACHE 环境变量在后续库导入前被设置
from config import config

from funasr import AutoModel
from modelscope.pipelines import pipeline
from core.device_utils import get_device

class EngineLoader:
    def __init__(self):
        self.asr_model = None
        self.vad_model = None
        self.streaming_model = None
        self.punc_model = None
        self.diarization_pipeline = None
        self.sv_pipeline = None

    def load_all(self):
        """加载所有核心模型"""
        logger.info("开始加载模型...")
        self._load_vad()
        self._load_asr()
        if config.enable_streaming:
            self._load_streaming()
        if config.enable_punc:
            self._load_punc()
        self._load_diarization()
        self._load_sv()
        logger.info("所有模型加载完毕。")

    def _load_streaming(self):
        """加载实时流式 ASR 模型 (如 Paraformer-zh-streaming)"""
        logger.info(f"加载流式 ASR 模型: {config.streaming_model}")
        device = get_device()
        try:
            self.streaming_model = AutoModel(
                model=config.streaming_model,
                disable_pbar=True,
                disable_update=True,
                device=device
            )
            logger.info("流式 ASR 模型加载成功。")
        except Exception as e:
            logger.error(f"流式 ASR 模型加载失败: {e}")
            self.streaming_model = None

    def _load_punc(self):
        """加载标点恢复模型 (如 CT-Transformer)"""
        logger.info(f"加载标点恢复模型: {config.punc_model}")
        device = get_device()
        try:
            self.punc_model = AutoModel(
                model=config.punc_model,
                disable_pbar=True,
                disable_update=True,
                device=device
            )
            logger.info("标点恢复模型加载成功。")
        except Exception as e:
            logger.error(f"标点恢复模型加载失败: {e}")
            self.punc_model = None

    def _load_vad(self):
        """加载 FSMN-VAD 模型"""
        logger.info(f"加载 VAD 模型: {config.vad_model}")
        device = get_device()
        # 遵循 FunASR 1.4.16 规范，在实例化 AutoModel 时绑定 device
        self.vad_model = AutoModel(
            model=config.vad_model,
            model_revision=config.vad_model_revision,
            disable_pbar=True,
            disable_update=True,
            device=device,
            max_end_silence_time=config.vad_max_end_silence_time,
            speech_noise_thres=config.vad_speech_noise_thres
        )
        logger.info(f"VAD 模型加载成功 (设备: {device})。")

    def _load_asr(self):
        """加载 SenseVoice 模型"""
        logger.info(f"加载 ASR 模型: {config.asr_model}")
        device = get_device()
        logger.info(f"ASR 模型将使用设备: {device}")
        # SenseVoiceSmall (AutoModel)
        self.asr_model = AutoModel(
            model=config.asr_model,
            trust_remote_code=True,
            disable_update=True,
            disable_pbar=True,
            device=device
        )
        logger.info("ASR 模型加载成功。")

    def _load_diarization(self):
        """加载 说话人分离 (Speaker Diarization) 模型"""
        logger.info(f"加载 Diarization 模型: {config.speaker_diarization_model}")
        device = get_device()
        try:
            self.diarization_pipeline = pipeline(
                task='speaker-diarization',
                model=config.speaker_diarization_model,
                device=device
            )
            logger.info(f"Diarization 模型加载成功 (设备: {device})。")
        except Exception as e:
            logger.error(f"Diarization 模型加载失败: {e}")
            self.diarization_pipeline = None

    def _load_sv(self):
        """加载 说话人验证特征提取 (Speaker Verification) 模型"""
        device = get_device()
        try:
            # eres2netv2_sv 需要使用 pipeline 加载
            self.sv_pipeline = pipeline(
                task='speaker-verification',
                model=config.speaker_sv_model,
                model_revision=config.speaker_sv_revision,
                device=device
            )
            logger.info(f"SV 模型加载成功 (设备: {device})。")
        except Exception as e:
            logger.error(f"SV 模型加载失败: {e}")
            self.sv_pipeline = None

engine_loader = EngineLoader()
