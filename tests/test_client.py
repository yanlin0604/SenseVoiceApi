import asyncio
import websockets
import json

async def run_test():
    uri = "ws://127.0.0.0:8000/ws/asr?mode=2"
    
    print(f"尝试连接 {uri}")
    async with websockets.connect(uri) as websocket:
        print("连接成功！")
        
        # 1. 构造一个假的 16kHz PCM 音频数据 (这里用全 0 模拟静音数据，真实情况应读取wav)
        # 3秒的音频数据 = 16000 * 2(16bit) * 3 = 96000 bytes
        fake_audio = b'\x00' * 96000
        
        print("正在发送音频数据...")
        await websocket.send({"bytes": fake_audio})
        
        # 2. 发送 flush 信号，要求立即识别
        print("发送 flush 信号...")
        await websocket.send(json.dumps({"text": "flush"}))
        
        # 这里只是模拟客户端发数据，服务端识别后不会将结果推给客户端，
        # 而是直接 POST 给 Java 的 Webhook (8080 端口)。
        # 因此，查看识别结果需要观察 mock_java_server 的输出。
        print("测试数据已发送完成。")

if __name__ == "__main__":
    asyncio.run(run_test())
