import uuid
import io
import asyncio
import torchaudio
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, Form, UploadFile, HTTPException
from loguru import logger
import uvicorn

from config import config
from models.engine_loader import engine_loader
from core.audio_pipeline import AudioPipeline
from core.speaker_embedding import extract_speaker_embedding
from core.text_postprocess import normalize_asr_text
from core.voiceprint_store import VoiceprintProfile, get_voiceprint_store

from fastapi.middleware.cors import CORSMiddleware

# 初始化应用
app = FastAPI(title="Medical ASR Service", version="1.0.0")

# 配置 CORS，允许所有来源的 HTTP 请求（解决离线上传等接口跨域报错）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("正在启动 Medical ASR Service...")
    # 同步加载模型可能会阻塞事件循环，但在启动时加载是推荐的做法
    engine_loader.load_all()

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("服务关闭，释放资源...")

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "Medical ASR Service is running",
        "streaming_enabled": config.enable_streaming,
        "asr_model": config.asr_model,
        "is_2pass": config.is_2pass_mode,
        "voiceprint_store": config.voiceprint_store_backend,
    }

@app.get("/api/models")
async def list_models():
    """返回当前模型配置和支持的模型列表"""
    return {
        "streaming_enabled": config.enable_streaming,
        "streaming_model": config.streaming_model if config.enable_streaming else None,
        "asr_model": config.asr_model,
        "is_2pass": config.is_2pass_mode,
        "streaming_chunk_size": config.parsed_chunk_size,
        "available_models": [
            {"name": "iic/SenseVoiceSmall", "type": "offline", "description": "170x 实时，情感/事件标签，CPU 可用"},
            {"name": "FunAudioLLM/Fun-ASR-Nano-2512", "type": "offline", "description": "LLM-based，31 语种，最高精度"},
            {"name": "Qwen/Qwen3-ASR-0.6B", "type": "offline", "description": "LLM-based，52 语种"},
            {"name": "paraformer-zh", "type": "offline", "description": "成熟中文 ASR，字级时间戳"},
            {"name": "paraformer-zh-streaming", "type": "streaming", "description": "流式中文 ASR"},
        ]
    }

@app.post("/api/offline-asr")
async def offline_asr(audio_file: UploadFile = File(...)):
    """
    接收完整的音频文件，直接对整段音频进行识别（不分离说话人）
    使用 VAD 切割长音频以防止 SenseVoice 截断。
    """
    try:
        audio_bytes = await audio_file.read()
        logger.info(f"开始对音频文件 {audio_file.filename} 进行整段离线识别...")
        
        # 加载音频（支持 torchaudio + pydub 兜底）并重采样到 16000Hz
        try:
            waveform, sample_rate = load_audio_to_waveform(audio_bytes, audio_file.filename)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
            
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000
            
        wav_np = waveform[0].numpy()
        
        # 使用 VAD 切分音频避免长音频截断
        vad_res = engine_loader.vad_model.generate(input=wav_np)
        
        full_text = ""
        if vad_res and len(vad_res) > 0 and "value" in vad_res[0]:
            segments = vad_res[0]["value"] # [[start_ms, end_ms], ...]
            for start_ms, end_ms in segments:
                start_frame = int((start_ms / 1000.0) * sample_rate)
                end_frame = int((end_ms / 1000.0) * sample_rate)
                sliced_wav = waveform[0:1, start_frame:end_frame]
                wav_np_seg = sliced_wav.numpy()[0]
                
                if len(wav_np_seg) > 400:
                    asr_res = await asyncio.to_thread(engine_loader.asr_model.generate, input=wav_np_seg)
                    if asr_res and len(asr_res) > 0 and "text" in asr_res[0]:
                        full_text += asr_res[0]["text"]

        # 统一后处理：清洗标签 + 数字归一化 + 标点补全
        punc_model = engine_loader.punc_model if config.enable_punc else None
        full_text = await asyncio.to_thread(normalize_asr_text, full_text, punc_model)

        logger.info(f"整段识别完成，最终文本长度: {len(full_text)}")
        return {
            "code": 200,
            "message": "success",
            "filename": audio_file.filename,
            "text": full_text
        }
    except Exception as e:
        logger.error(f"整段识别失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/offline-diarize")
async def offline_diarize(audio_file: UploadFile = File(...)):
    """
    接收完整的音频文件，使用 CampPlus 进行离线的整段说话人分离
    """
    if not engine_loader.diarization_pipeline:
        raise HTTPException(status_code=500, detail="Diarization 模型未加载或不支持")
        
    try:
        # 读取上传的音频文件内容
        audio_bytes = await audio_file.read()
        
        # 加载音频（支持 torchaudio + pydub 兜底）并重采样到 16000Hz
        try:
            waveform, sample_rate = load_audio_to_waveform(audio_bytes, audio_file.filename)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        
        # 统一重采样到 16000Hz（SenseVoice 和 CampPlus 都需要 16k）
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000
            
        wav_np = waveform[0].numpy()
        
        # 调用说话人分离模型进行处理，传入 16k numpy 数组
        logger.info(f"开始对音频文件 {audio_file.filename} 进行离线说话人分离 (已重采样至 16kHz)...")
        diarize_res = await asyncio.to_thread(
            engine_loader.diarization_pipeline,
            wav_np, 
            sample_rate=sample_rate
        )
        
        if not diarize_res or not isinstance(diarize_res, dict) or "text" not in diarize_res:
            return {"code": 200, "message": "success", "filename": audio_file.filename, "data": []}
            
        segments = diarize_res["text"]
        logger.info(f"分离完成, 共发现 {len(segments)} 个片段。开始分段识别文字...")
        
        final_results = []
        for start_sec, end_sec, spk in segments:
            # 切片音频
            start_frame = int(start_sec * sample_rate)
            end_frame = int(end_sec * sample_rate)
            sliced_wav = waveform[0:1, start_frame:end_frame]
            
            # 转换为 numpy 数组送入 ASR 引擎
            wav_np_seg = sliced_wav.numpy()[0]
            
            # 由于可能切出的音频极短，加一层容错
            text = ""
            if len(wav_np_seg) > 400:  # 避免过短音频报错
                asr_res = await asyncio.to_thread(
                    engine_loader.asr_model.generate,
                    input=wav_np_seg
                )
                if asr_res and len(asr_res) > 0 and "text" in asr_res[0]:
                    text = asr_res[0]["text"]

            # 每段做后处理：清洗标签 + 数字归一化 + 标点
            if text:
                punc_model = engine_loader.punc_model if config.enable_punc else None
                text = await asyncio.to_thread(normalize_asr_text, text, punc_model)

            final_results.append({
                "speaker": str(spk),
                "start": float(start_sec),
                "end": float(end_sec),
                "text": text
            })
            
        logger.info(f"文件 {audio_file.filename} 的分离+识别处理完毕。")
        
        return {
            "code": 200,
            "message": "success",
            "filename": audio_file.filename,
            "data": final_results
        }
    except Exception as e:
        logger.error(f"离线说话人分离与识别失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/voiceprints/enroll")
async def enroll_voiceprint(
    audio_file: UploadFile = File(...),
    profile_id: str = Form(...),
    doctor_code: str = Form(...),
    speaker_name: str = Form(...),
    speaker_title: str = Form(""),
    hospital_code: str = Form("default"),
    dept_code: str = Form(""),
    match_threshold: Optional[float] = Form(None),
    is_active: bool = Form(True),
):
    """
    注册或覆盖医生声纹。业务元数据由后端管理，ASR 服务负责提取向量并写入声纹库。
    """
    try:
        wav_np = await _load_audio_as_16k_mono(audio_file)
        embedding = await extract_speaker_embedding(wav_np)
        threshold = match_threshold or config.doctor_voiceprint_match_threshold
        profile = VoiceprintProfile(
            profile_id=profile_id,
            doctor_code=doctor_code,
            speaker_name=speaker_name,
            speaker_title=speaker_title,
            hospital_code=hospital_code or "default",
            dept_code=dept_code or "",
            match_threshold=float(threshold),
            is_active=bool(is_active),
        )
        get_voiceprint_store().upsert(profile, embedding)
        logger.info(f"医生声纹注册完成: profileId={profile_id}, doctorCode={doctor_code}, speaker={profile.display_label}")
        return {
            "code": 200,
            "message": "success",
            "data": {
                "vectorId": profile_id,
                "doctorCode": doctor_code,
                "speakerName": speaker_name,
                "speakerTitle": speaker_title,
                "displayLabel": profile.display_label,
                "embeddingDim": int(len(embedding)),
                "voiceprintVersion": config.speaker_sv_revision,
                "matchThreshold": float(threshold),
                "storeBackend": config.voiceprint_store_backend,
            },
        }
    except Exception as e:
        logger.error(f"医生声纹注册失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/voiceprints/match")
async def match_voiceprint(
    audio_file: UploadFile = File(...),
    hospital_code: str = Form("default"),
    dept_code: str = Form(""),
):
    """
    调试用声纹匹配接口，用于后台录入后校验命中结果。
    """
    try:
        wav_np = await _load_audio_as_16k_mono(audio_file)
        embedding = await extract_speaker_embedding(wav_np)
        match = get_voiceprint_store().search(embedding, hospital_code=hospital_code, dept_code=dept_code)
        if not match:
            return {"code": 200, "message": "success", "data": {"matched": False}}
        profile = match.profile
        return {
            "code": 200,
            "message": "success",
            "data": {
                "matched": True,
                "vectorId": profile.profile_id,
                "doctorCode": profile.doctor_code,
                "speakerName": profile.speaker_name,
                "speakerTitle": profile.speaker_title,
                "displayLabel": profile.display_label,
                "score": match.score,
            },
        }
    except Exception as e:
        logger.error(f"医生声纹匹配失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def load_audio_to_waveform(audio_bytes: bytes, filename: Optional[str] = None):
    """
    尝试从字节流中加载音频。如果 torchaudio.load 失败，则使用 pydub 兜底解析。
    返回 (waveform, sample_rate)
    """
    try:
        return torchaudio.load(io.BytesIO(audio_bytes))
    except Exception as e:
        logger.warning(f"torchaudio.load 加载音频失败 ({e})，尝试使用 pydub 兜底解析...")
        try:
            from pydub import AudioSegment
            ext = None
            if filename:
                ext_parts = filename.split('.')
                if len(ext_parts) > 1:
                    ext = ext_parts[-1].lower()
            
            # 使用 pydub 读取，并导出为标准 wav 格式供 torchaudio 读取
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=ext)
            wav_io = io.BytesIO()
            audio.export(wav_io, format="wav")
            wav_io.seek(0)
            return torchaudio.load(wav_io)
        except Exception as pe:
            logger.error(f"pydub 解析音频也失败: {pe}")
            raise ValueError(f"不支持的音频文件格式或文件损坏: {pe}")


async def _load_audio_as_16k_mono(audio_file: UploadFile):
    audio_bytes = await audio_file.read()
    filename = audio_file.filename or "voiceprint.wav"
    logger.info(f"[_load_audio_as_16k_mono] 收到声纹音频注册/匹配请求. filename={filename}, size={len(audio_bytes)} bytes, content_type={audio_file.content_type}")
    if len(audio_bytes) > 8:
        logger.info(f"[_load_audio_as_16k_mono] 音频前8字节: {audio_bytes[:8].hex()}")
    
    try:
        waveform, sample_rate = load_audio_to_waveform(audio_bytes, filename)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
        
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    wav_np = waveform[0].numpy()
    if len(wav_np) < 16000:
        raise HTTPException(status_code=400, detail="声纹录入音频过短，至少需要约 1 秒有效语音")
    return wav_np


@app.websocket("/ws/asr")
async def asr_websocket(websocket: WebSocket, mode: int = 1, hospital_code: str = None, dept_code: str = None):
    """
    WebSocket 音频接收端点。
    :param mode: 1 为纯语音识别，2 为说话人分离。可通过 query param 指定 /ws/asr?mode=2
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info(f"客户端连接成功, Session ID: {session_id}, Mode: {mode}")
    
    pipeline = AudioPipeline(session_id=session_id, mode=mode, hospital_code=hospital_code, dept_code=dept_code)
    
    try:
        while True:
            # 接收客户端发来的音频字节数据 (bytes)
            # 或者客户端可以发 JSON，里面包含控制指令如 {"action": "flush"}
            message = await websocket.receive()
            
            if "bytes" in message:
                audio_bytes = message["bytes"]
                # 将收到的音频块送入管道处理（并传入 websocket 以便 VAD 自动断句时能回传结果）
                await pipeline.process_chunk(audio_bytes, websocket)
                
            elif "text" in message:
                text_data = message["text"]
                if text_data == "flush":
                    logger.debug(f"[{session_id}] 收到前端 flush 信号，强制送识")
                    await pipeline.flush(websocket)
                else:
                    logger.warning(f"[{session_id}] 收到未知的文本指令: {text_data}")

    except WebSocketDisconnect:
        await pipeline.flush(websocket=websocket)
        logger.info(f"客户端断开连接 [{session_id}]")
    except RuntimeError as e:
        if "Cannot call" in str(e) and "disconnect" in str(e):
            await pipeline.flush(websocket=websocket)
            logger.info(f"客户端正常断开连接 [{session_id}]")
        else:
            logger.error(f"WebSocket RuntimeError [{session_id}]: {e}")
            await pipeline.flush(websocket=websocket)
    except Exception as e:
        logger.error(f"WebSocket 异常 [{session_id}]: {e}")
        await pipeline.flush(websocket=websocket)

if __name__ == "__main__":
    logger.info(f"服务将启动于 {config.host}:{config.port}")
    uvicorn.run(
        "main:app", 
        host=config.host, 
        port=config.port, 
        reload=False,
        ws_ping_interval=None,  # 禁用心跳，防止 CPU 被大模型占满时误判超时断开
        ws_ping_timeout=None
    )
