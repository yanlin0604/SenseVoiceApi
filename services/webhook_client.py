import httpx
from loguru import logger
from config import config

class WebhookClient:
    def __init__(self):
        self.client = httpx.AsyncClient()
        self.url = config.webhook_url

    async def push_to_java(self, session_id: str, text: str, speaker: str = None, is_final: bool = True):
        """
        推送识别结果到 Java 后端
        """
        payload = {
            "session_id": session_id,
            "text": text,
            "is_final": is_final
        }
        if speaker:
            payload["speaker"] = speaker
            
        try:
            response = await self.client.post(self.url, json=payload)
            response.raise_for_status()
            logger.debug(f"Webhook 推送成功: {payload}")
        except Exception as e:
            logger.error(f"Webhook 推送失败: {e} | payload={payload}")

    async def close(self):
        await self.client.aclose()

webhook_client = WebhookClient()
