import os
import torch
from loguru import logger

# 必须先引入 config，以确保 MODELSCOPE_CACHE 环境变量在后续库导入前被设置
from config import config

from funasr import AutoModel
from modelscope.pipelines import pipeline

class EngineLoader:
    def __init__(self):
        self.asr_model = None
        self.vad_model = None
        self.streaming_model = None  # 流式 ASR 模型 (Paraformer-zh-streaming)
        self.punc_model = None       # 标点模型 (ct-punc)
        self.diarization_pipeline = None
        self.sv_pipeline = None

    def load_all(self):
        """加载所有核心模型"""
        logger.info("开始加载模型...")
        self._load_vad()
        self._load_streaming()
        self._load_asr()
        self._load_punc()
        self._load_diarization()
        self._load_sv()
        logger.info("所有模型加载完毕。")
        self._log_model_summary()

    def _load_vad(self):
        """加载 FSMN-VAD 模型"""
        logger.info(f"加载 VAD 模型: {config.vad_model}")
        # Note: AutoModel会自动下载如果本地不存在
        self.vad_model = AutoModel(
            model=config.vad_model,
            model_revision=config.vad_model_revision,
            disable_pbar=True,
            disable_update=True,
            max_end_silence_time=config.vad_max_end_silence_time,
            speech_noise_thres=config.vad_speech_noise_thres
        )
        logger.info("VAD 模型加载成功。")

    def _load_streaming(self):
        """加载流式 ASR 模型 (Paraformer-zh-streaming)"""
        if not config.enable_streaming:
            logger.info("流式识别未启用 (ENABLE_STREAMING=false)，跳过加载流式模型。")
            return
        
        logger.info(f"加载流式 ASR 模型: {config.streaming_model}")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"流式模型将使用设备: {device}")
        
        self.streaming_model = AutoModel(
            model=config.streaming_model,
            device=device,
            disable_update=True,
            disable_pbar=True,
        )
        logger.info("流式 ASR 模型加载成功。")

    def _load_asr(self):
        """
        加载精修/离线 ASR 模型。
        支持的模型：SenseVoiceSmall, Fun-ASR-Nano, Qwen3-ASR, Paraformer-zh, Paraformer-zh-streaming
        """
        model_name = config.asr_model
        
        # 如果精修模型和流式模型相同，直接复用已加载的流式模型
        if config.enable_streaming and model_name == config.streaming_model:
            if self.streaming_model is not None:
                self.asr_model = self.streaming_model
                logger.info(f"精修模型与流式模型相同 ({model_name})，复用已加载的流式模型。")
                return
        
        logger.info(f"加载精修 ASR 模型: {model_name}")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"精修 ASR 模型将使用设备: {device}")
        
        # 构建 AutoModel 参数（不同模型需要不同的参数组合）
        kwargs = {
            "model": model_name,
            "device": device,
            "disable_update": True,
            "disable_pbar": True,
        }
        
        # Qwen3-ASR 需要 bf16 精度和 hub 配置
        if "Qwen3-ASR" in model_name or "qwen3" in model_name.lower():
            kwargs["dtype"] = "bf16"
            kwargs["hub"] = "ms"
            logger.info("检测到 Qwen3-ASR 模型，启用 bf16 精度并设置 hub 为 ms。")
        
        # SenseVoiceSmall / Fun-ASR-Nano 需要 trust_remote_code
        if "SenseVoice" in model_name or "Fun-ASR-Nano" in model_name:
            kwargs["trust_remote_code"] = True
        
        self.asr_model = AutoModel(**kwargs)
        logger.info(f"精修 ASR 模型加载成功: {model_name}")

    def _load_punc(self):
        """加载标点模型 (ct-punc)，给离线/2pass 精修结果补标点。CPU 可用。"""
        if not config.enable_punc:
            logger.info("标点功能未启用 (ENABLE_PUNC=false)，跳过加载标点模型。")
            return
        logger.info(f"加载标点模型: {config.punc_model}")
        try:
            self.punc_model = AutoModel(
                model=config.punc_model,
                disable_update=True,
                disable_pbar=True,
            )
            logger.info("标点模型加载成功。")
        except Exception as e:
            logger.error(f"标点模型加载失败，将以无标点模式运行: {e}")
            self.punc_model = None

    def _load_diarization(self):
        """加载 说话人分离 (Speaker Diarization) 模型"""
        logger.info(f"加载 Diarization 模型: {config.speaker_diarization_model}")
        try:
            self.diarization_pipeline = pipeline(
                task='speaker-diarization',
                model=config.speaker_diarization_model
            )
            logger.info("Diarization 模型加载成功。")
        except Exception as e:
            logger.error(f"Diarization 模型加载失败: {e}")
            self.diarization_pipeline = None

    def _load_sv(self):
        """加载 说话人验证特征提取 (Speaker Verification) 模型"""
        try:
            # eres2netv2_sv 需要使用 pipeline 加载
            self.sv_pipeline = pipeline(
                task='speaker-verification',
                model=config.speaker_sv_model,
                model_revision=config.speaker_sv_revision
            )
            logger.info("SV 模型加载成功。")
        except Exception as e:
            logger.error(f"SV 模型加载失败: {e}")
            self.sv_pipeline = None

    def _log_model_summary(self):
        """输出当前模型加载状态摘要"""
        logger.info("=" * 60)
        logger.info("模型加载摘要:")
        logger.info(f"  运行设备: {'cuda:0 (GPU)' if torch.cuda.is_available() else 'cpu (无GPU)'}")
        logger.info(f"  VAD 模型: {config.vad_model} ✓")
        logger.info(f"  流式模式: {'✓ 已启用' if config.enable_streaming else '✗ 未启用'}")
        if config.enable_streaming:
            logger.info(f"  流式模型: {config.streaming_model} ✓")
            logger.info(f"  chunk 配置: {config.parsed_chunk_size}")
        logger.info(f"  精修模型: {config.asr_model} ✓")
        logger.info(f"  2pass 模式: {'✓' if config.is_2pass_mode else '✗'}")
        logger.info(f"  标点模型: {'✓ ' + config.punc_model if self.punc_model else '✗ 未启用'}")
        logger.info(f"  数字归一化(ITN): {'✓' if config.enable_itn else '✗'}")
        logger.info(f"  Diarization: {'✓' if self.diarization_pipeline else '✗'}")
        logger.info(f"  Speaker Verification: {'✓' if self.sv_pipeline else '✗'}")
        logger.info("=" * 60)

engine_loader = EngineLoader()
