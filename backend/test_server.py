"""
简单的测试服务器
用于验证FastAPI基本功能
"""

from fastapi import FastAPI
import uvicorn

app = FastAPI(title="测试服务器")

@app.get("/")
async def root():
    return {"message": "测试服务器运行正常"}

@app.get("/health")
async def health():
    return {"status": "healthy", "message": "服务器运行正常"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info") 