# SenseVoiceApi (Medical ASR Service)

基于 FastAPI 和 FunASR 构建的医疗语音识别与字段实时提取服务。

## 简介
本项目提供高性能的语音转文本 (ASR) 能力，结合双向 WebSocket 通信，可实现实时的医疗问诊语音识别，并与 Java 业务后台协同完成病历字段的智能提取与填充。

## 核心特性
- **实时语音识别**: 采用 WebSocket 提供实时的音频流转写。
- **高性能模型**: 集成 FunASR / SenseVoice 模型，提供高准确率的识别能力。
- **架构设计**: 负责核心的语音转文本任务，并通过专门的通道与前端及 Java 后台协同，实现无缝的医疗问诊工作流。

## 技术栈
- Python (FastAPI, Uvicorn, WebSockets)
- FunASR, ModelScope, PyTorch

## 安装与运行

1. 安装依赖:
```bash
pip install -r requirements.txt
```

2. 运行服务:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
