import os
import json
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Server API config
    host: str = "0.0.0.0"
    port: int = 8000
    
    # 精修/离线 ASR 模型（5 选 1）:
    #   iic/SenseVoiceSmall           - 170x 实时，情感/事件标签，CPU 可用
    #   FunAudioLLM/Fun-ASR-Nano-2512 - LLM-based，31 语种，最高精度（需 GPU）
    #   Qwen/Qwen3-ASR-0.6B           - LLM-based，52 语种（需安装 qwen-asr，需 GPU）
    #   paraformer-zh                  - 成熟中文 ASR，字级时间戳
    #   paraformer-zh-streaming        - 流式模型（与流式模型相同时跳过二次识别）
    asr_model: str = "iic/SenseVoiceSmall"
    
    # VAD 模型配置
    vad_model: str = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
    vad_model_revision: str = "v2.0.4"
    
    # 流式识别配置
    enable_streaming: bool = False
    streaming_model: str = "paraformer-zh-streaming"
    # 流式 chunk 配置 [lookback, chunk, lookahead]，单位 60ms
    # [0, 10, 5] = 600ms 延迟, [0, 8, 4] = 480ms 延迟
    streaming_chunk_size: str = "[0, 8, 4]"
    streaming_encoder_chunk_look_back: int = 4
    streaming_decoder_chunk_look_back: int = 1
    
    # 标点模型配置（给离线/2pass 精修结果补标点；流式中间结果不加标点以保证低延迟）
    # 注意：SenseVoice 自带标点，应设为 False 避免重复标点
    enable_punc: bool = False
    punc_model: str = "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
    # 中文数字 → 阿拉伯数字归一化（医疗剂量/年份场景，如"五毫克"→"5毫克"）
    enable_itn: bool = True

    # 说话人分离与声纹验证配置
    speaker_diarization_model: str = "iic/speech_campplus_speaker-diarization_common"
    speaker_sv_model: str = "iic/speech_eres2netv2_sv_zh-cn_16k-common"
    speaker_sv_revision: str = "v1.0.2"
    
    # 全局模型缓存与下载目录 (默认存在当前项目的 models_cache 文件夹)
    model_cache_dir: str = "./models_cache"
    
    # 声纹比对相似度阈值
    sv_similarity_threshold: float = 0.6
    
    # Audio slicing & VAD param
    vad_max_end_silence_time: int = 800
    vad_speech_noise_thres: float = 0.8
    
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        env_file_encoding='utf-8',
        extra='ignore'
    )
    
    @property
    def parsed_chunk_size(self) -> list:
        """将字符串格式的 chunk_size 解析为 Python list"""
        try:
            return json.loads(self.streaming_chunk_size)
        except (json.JSONDecodeError, TypeError):
            return [0, 10, 5]
    
    @property
    def is_2pass_mode(self) -> bool:
        """判断是否为 2pass 模式（流式 + 不同的精修模型）"""
        return self.enable_streaming and self.asr_model != self.streaming_model

config = Settings()

# 在导入任何模型库之前，设置 ModelScope 的全局缓存环境变量
# 这样如果模型不存在，就会自动下载并存放到这个自定义的文件夹里
if config.model_cache_dir:
    os.environ["MODELSCOPE_CACHE"] = os.path.abspath(config.model_cache_dir)
