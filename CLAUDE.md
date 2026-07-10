[根目录](../CLAUDE.md) > **medical-asr-service**

# medical-asr-service — 医疗实时语音识别服务（SenseVoiceApi）

> 文档生成时间：2026-07-09 15:40:12

## 变更记录 (Changelog)

| 时间 | 变更内容 |
|------|---------|
| 2026-07-09 15:40:12 | 初始生成模块级 CLAUDE.md |

---

## 模块职责

基于 FastAPI 的高性能**双向流式实时 ASR 核心服务**，为 medai-plugin 与 MedAi-Service 提供语音能力：

- **2-pass 实时识别**：第一遍流式极速输出中间结果（Partial），第二遍端点/句末精修纠错与标点（Final）
- **多模型支持**：SenseVoiceSmall（CPU 友好）、Fun-ASR-Nano、Qwen3-ASR-0.6B、paraformer-zh(-streaming)
- **说话人分离与医生声纹**：CAM++ diarization + ERes2NetV2 声纹验证；医生声纹注册/检索/实时匹配（“上一位已识别医生优先”防跳变策略）
- **文本后处理**：语气词过滤、智能断句、标点、中文数字→阿拉伯数字（ITN，医疗剂量场景）
- **Webhook 解耦**：`is_final=true` 结果可推送下游业务系统

## 入口与启动

| 项 | 位置 |
|----|------|
| 入口 | `main.py`（FastAPI app，启动时 `engine_loader.load_all()` 预载模型） |
| 配置 | `config.py`（pydantic-settings，读 `.env`） |
| 启动 | `uvicorn main:app --host 0.0.0.0 --port 8000` |

## 对外接口

- `WS /ws/asr`：核心流式识别（16kHz/16bit/mono PCM 二进制入，JSON `{is_final, text, speaker, timestamp}` 出，含声纹匹配的医生姓名/职称）
- `GET /health`、`GET /api/models`：健康与模型信息
- `POST /api/offline-asr`：整段音频离线识别（VAD 切割防截断）
- `POST /api/voiceprints/enroll`、`POST /api/voiceprints/match`：医生声纹注册与检索（由 MedAi-Service 的 `AsrVoiceprintClient` 调用）

## 关键依赖与配置

- FastAPI + uvicorn + loguru + torchaudio + FunASR/ModelScope 模型栈
- 声纹库：Milvus（`192.168.2.43:19530`，库 `vocal_print`，集合 `medai_doctor_voiceprints`），连接失败自动降级本地 JSON（`models_cache/doctor_voiceprints.json`）
- 关键阈值：`sv_similarity_threshold=0.6`、`doctor_voiceprint_match_threshold=0.85`
- 模型缓存：`models_cache/`（**大文件目录，勿扫描**）

## 数据模型

- `core/voiceprint_store.py`：`VoiceprintProfile`（医生声纹档案 + embedding 向量）
- `core/audio_pipeline.py`：`AudioPipeline` 实时管线（VAD 切片 → 流式/离线识别 → 声纹身份判断 → 后处理）
- `core/speaker_embedding.py`：说话人 embedding 提取
- `models/engine_loader.py`：多模型统一加载器

## 测试与质量

- `tests/`：`mock_java_server`（模拟 Java 后端 Webhook 接收）、`test.html`（浏览器手工联调页）
- 无 pytest 自动化用例；`.spec-workflow/` 为规格工作流模板

## 常见问题 (FAQ)

- **无 GPU 能跑吗？** 默认模型 `iic/SenseVoiceSmall` CPU 可用；Qwen3-ASR/Fun-ASR 需 GPU。
- **没有 Milvus？** 自动降级本地 JSON 存储，`/health` 返回 `voiceprint_store` 当前后端。

## 相关文件清单

- `main.py`、`config.py`、`requirements.txt`、`.env`
- `core/{audio_pipeline,speaker_embedding,voiceprint_store,text_postprocess}.py`
- `models/engine_loader.py`
- `docs/superpowers/specs/2026-06-23-realtime-field-extraction-design.md`（实时字段提取设计）
