import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Server API config
    host: str = "0.0.0.0"
    port: int = 8000

    # Webhook config
    # 是否推送识别结果到 Java 后端 (ENABLE_WEBHOOK=false 时跳过推送，适合单独调试 ASR 服务)
    enable_webhook: bool = False
    webhook_url: str = "http://localhost:8080/api/medical/asr/callback"

    asr_model: str = "iic/SenseVoiceSmall"
    asr_language: str = "zh"
    asr_use_itn: bool = True
    asr_ban_emo_unk: bool = True
    asr_hotwords: str = ""

    # 流式识别配置 (Paraformer 实时出字层)
    enable_streaming: bool = False
    streaming_model: str = "paraformer-zh-streaming"
    streaming_chunk_size: str = "[0, 10, 5]"
    streaming_encoder_chunk_look_back: int = 4
    streaming_decoder_chunk_look_back: int = 1

    # 标点与数字归一化配置
    enable_punc: bool = False
    punc_model: str = "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
    enable_itn: bool = True

    vad_model: str = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
    vad_model_revision: str = "v2.0.4"
    speaker_diarization_model: str = "iic/speech_campplus_speaker-diarization_common"
    speaker_sv_model: str = "iic/speech_eres2net_sv_zh-cn_16k-common"
    speaker_sv_revision: str = "v1.0.5"

    # 全局模型缓存与下载目录 (默认存在当前项目的 models_cache 文件夹)
    model_cache_dir: str = "./models_cache"

    # 声纹比对相似度阈值
    sv_similarity_threshold: float = 0.6

    # Audio slicing & VAD param
    vad_max_end_silence_time: int = 800
    vad_speech_noise_thres: float = 0.8
    chunk_size_ms: int = 200

    # 计算设备配置: auto (默认自动检测，优先GPU，失败降级CPU), cpu, cuda:0
    device: str = "auto"

    # 并发推理控制 (Semaphore 并发限制，防止打爆 GPU 显存)
    max_concurrent_vad: int = 4
    max_concurrent_asr: int = 2
    max_concurrent_sv: int = 2

    # 纯本地声纹管理配置 (100% 纯本地化，无外部云厂商依赖)
    local_voiceprint_dir: str = "voiceprint_db"
    local_voiceprint_threshold: float = 0.4
    default_group_id: str = "iFLYTEK_voiceprint_group"

    # Milvus 向量库配置 (可选，用于本地声纹高维向量索引与1:N极速检索)
    milvus_enabled: bool = False
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_database: str = "default"
    milvus_user: str = ""
    milvus_password: str = ""
    milvus_collection_prefix: str = "voiceprint_"

    # 第三方业务 API 地址 (可选)
    third_party_api_base_url: str = "http://localhost:8080"

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        env_file_encoding='utf-8',
        extra='ignore'
    )

config = Settings()

# 在导入任何模型库之前，设置 ModelScope 的全局缓存环境变量
# 这样如果模型不存在，就会自动下载并存放到这个自定义的文件夹里
if config.model_cache_dir:
    os.environ["MODELSCOPE_CACHE"] = os.path.abspath(config.model_cache_dir)
