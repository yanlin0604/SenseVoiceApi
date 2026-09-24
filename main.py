import uuid
import io
import os
import sys
import asyncio
from typing import Optional
import torchaudio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, HTTPException
from loguru import logger
import uvicorn

# 确保项目根目录在 sys.path 中，以便无缝导入 modules
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import config
from models.engine_loader import engine_loader
from core.audio_pipeline import AudioPipeline
from core.concurrency import limit_vad, limit_asr, limit_sv
from services.webhook_client import webhook_client
from routers.voiceprint import router as voiceprint_router
from modules.text_formatter import format_str_v3

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

# 挂载声纹管理路由
app.include_router(voiceprint_router)

@app.on_event("startup")
async def startup_event():
    logger.info("正在启动 Medical ASR Service...")
    # 同步加载模型可能会阻塞事件循环，但在启动时加载是推荐的做法
    engine_loader.load_all()
    # 预热声纹客户端单例，消除首个请求冷启动
    try:
        from routers.voiceprint import get_voiceprint_backend
        get_voiceprint_backend()
        logger.info("声纹服务客户端预热完成。")
    except Exception as e:
        logger.warning(f"声纹服务客户端预热警告: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("服务关闭，释放资源...")
    await webhook_client.close()

from fastapi.responses import FileResponse

@app.get("/")
@app.get("/test.html")
async def index():
    """返回测试工作台页面"""
    test_html = os.path.join(os.path.dirname(__file__), "tests", "test.html")
    if os.path.exists(test_html):
        return FileResponse(test_html)
    return {"status": "ok", "message": "Medical ASR Service is running"}

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Medical ASR Service is running"}

@app.post("/api/offline-asr")
async def offline_asr(audio_file: UploadFile = File(...)):
    """
    接收完整的音频文件，直接对整段音频进行识别（不分离说话人）
    使用 VAD 切割长音频以防止 SenseVoice 截断。
    """
    try:
        audio_bytes = await audio_file.read()
        logger.info(f"开始对音频文件 {audio_file.filename} 进行整段离线识别...")
        
        # 先用 torchaudio 读取并重采样到 16000Hz
        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000
            
        wav_np = waveform[0].numpy()
        
        # 使用 VAD 切分音频避免长音频截断 (信号量限流保护)
        async with limit_vad():
            vad_res = await asyncio.to_thread(engine_loader.vad_model.generate, input=wav_np)
        
        full_text = ""
        if vad_res and len(vad_res) > 0 and "value" in vad_res[0]:
            segments = vad_res[0]["value"] # [[start_ms, end_ms], ...]
            for start_ms, end_ms in segments:
                start_frame = int((start_ms / 1000.0) * sample_rate)
                end_frame = int((end_ms / 1000.0) * sample_rate)
                sliced_wav = waveform[0:1, start_frame:end_frame]
                wav_np_seg = sliced_wav.numpy()[0]
                
                if len(wav_np_seg) > 400:
                    gen_kwargs = {
                        "language": config.asr_language,
                        "use_itn": config.asr_use_itn,
                        "ban_emo_unk": config.asr_ban_emo_unk,
                    }
                    if config.asr_hotwords:
                        gen_kwargs["postprocess_hotwords"] = config.asr_hotwords

                    async with limit_asr():
                        asr_res = await asyncio.to_thread(
                            engine_loader.asr_model.generate,
                            input=wav_np_seg,
                            **gen_kwargs
                        )
                    if asr_res and len(asr_res) > 0 and "text" in asr_res[0]:
                        full_text += format_str_v3(asr_res[0]["text"])
        
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
        
        # 先用 torchaudio 读取音频为 waveform
        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
        
        # 统一重采样到 16000Hz（SenseVoice 和 CampPlus 都需要 16k）
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000
            
        wav_np = waveform[0].numpy()
        
        # 调用说话人分离模型进行处理，传入 16k numpy 数组 (信号量限流保护)
        logger.info(f"开始对音频文件 {audio_file.filename} 进行离线说话人分离 (已重采样至 16kHz)...")
        async with limit_sv():
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
                gen_kwargs = {
                    "language": config.asr_language,
                    "use_itn": config.asr_use_itn,
                    "ban_emo_unk": config.asr_ban_emo_unk,
                }
                if config.asr_hotwords:
                    gen_kwargs["postprocess_hotwords"] = config.asr_hotwords

                async with limit_asr():
                    asr_res = await asyncio.to_thread(
                        engine_loader.asr_model.generate,
                        input=wav_np_seg,
                        **gen_kwargs
                    )
                if asr_res and len(asr_res) > 0 and "text" in asr_res[0]:
                    text = format_str_v3(asr_res[0]["text"])
            
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

@app.websocket("/ws/asr")
async def asr_websocket(
    websocket: WebSocket,
    mode: int = 1,
    user_id: Optional[str] = None,
    userId: Optional[str] = None,
):
    """
    WebSocket 音频接收端点。
    :param mode: 
      - 1: 纯语音识别（不进行说话人分离）
      - 2: 会话内临时声纹盲分（区分“用户1”、“用户2”）
      - 3: 持久化声纹库实名识别 + 第三方资料联动（自动匹配姓名、头像、角色，未匹配时平滑回退）
      例如：/ws/asr?mode=3&user_id=101
    :param user_id: 用户唯一标识（传入时定向拉取该用户的候选声纹列表，进行小范围精准比对）
    :param userId: 兼容前端驼峰命名
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    effective_user_id = user_id or userId
    logger.info(f"客户端连接成功, Session ID: {session_id}, Mode: {mode}, User ID: {effective_user_id}")
    
    pipeline = AudioPipeline(session_id=session_id, mode=mode, user_id=effective_user_id)
    
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
    uvicorn.run("main:app", host=config.host, port=config.port, reload=False)
