from fastapi import FastAPI, Request
from loguru import logger
import uvicorn

app = FastAPI(title="Mock Java Server")

@app.post("/api/medical/asr/callback")
async def receive_asr_result(request: Request):
    payload = await request.json()
    logger.info(f"=========== 收到 Python 识别结果 ===========")
    logger.info(f"Session ID: {payload.get('session_id')}")
    logger.info(f"Speaker   : {payload.get('speaker', 'N/A')}")
    logger.info(f"Text      : {payload.get('text')}")
    logger.info(f"Is Final  : {payload.get('is_final')}")
    logger.info(f"==========================================")
    return {"code": 200, "message": "success"}

if __name__ == "__main__":
    logger.info("Mock Java Server running on port 8081...")
    uvicorn.run("mock_java_server:app", host="0.0.0.0", port=8081)
