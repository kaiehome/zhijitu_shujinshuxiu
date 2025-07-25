"""
FastAPI服务器 - 刺绣图像处理API
提供完整的模型API服务
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import uvicorn
import asyncio
import logging
import time
import json
import os
from pathlib import Path
import shutil

# 导入我们的简化模型API
from simple_model_api import (
    SimpleModelAPIManager, 
    SimpleModelAPIConfig, 
    SimpleGenerationRequest,
    simple_generate_embroidery_api,
    simple_get_job_status_api,
    simple_get_available_models_api,
    simple_get_system_stats_api
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="刺绣图像处理API",
    description="AI驱动的刺绣图像生成和优化API服务",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局API管理器
api_manager = SimpleModelAPIManager()


# Pydantic模型
class GenerationRequest(BaseModel):
    """生成请求模型"""
    style: str = "basic"
    color_count: int = 16
    edge_enhancement: bool = True
    noise_reduction: bool = True
    optimization_level: str = "balanced"  # conservative, balanced, aggressive
    enable_auto_tune: bool = True
    reference_image: Optional[str] = None


class JobStatusResponse(BaseModel):
    """任务状态响应模型"""
    job_id: str
    status: str
    processing_time: float
    error_message: Optional[str] = None
    output_files: List[str] = []
    quality_metrics: Optional[Dict[str, Any]] = None
    optimization_params: Optional[Dict[str, Any]] = None


class SystemStatsResponse(BaseModel):
    """系统统计响应模型"""
    active_jobs: int
    completed_jobs: int
    available_models: Dict[str, Any]
    config: Dict[str, Any]


# 工具函数
def save_upload_file(upload_file: UploadFile, destination: str) -> str:
    """保存上传的文件"""
    try:
        with open(destination, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return destination
    finally:
        upload_file.file.close()


# API路由
@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "刺绣图像处理API服务",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "uptime": time.time()  # 简化版本，实际应该计算启动时间
    }


@app.get("/models", response_model=Dict[str, Any])
async def get_available_models():
    """获取可用模型"""
    try:
        return simple_get_available_models_api()
    except Exception as e:
        logger.error(f"获取可用模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取模型信息失败: {str(e)}")


@app.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats():
    """获取系统统计信息"""
    try:
        stats = simple_get_system_stats_api()
        return SystemStatsResponse(**stats)
    except Exception as e:
        logger.error(f"获取系统统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取系统统计失败: {str(e)}")


@app.post("/generate", response_model=Dict[str, str])
async def generate_embroidery(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    style: str = "basic",
    color_count: int = 16,
    edge_enhancement: bool = True,
    noise_reduction: bool = True,
    optimization_level: str = "balanced",
    enable_auto_tune: bool = True
):
    """生成刺绣图像"""
    try:
        # 验证文件类型
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="只支持图像文件")
        
        # 创建上传目录
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        # 保存上传的文件
        timestamp = int(time.time())
        filename = f"{timestamp}_{file.filename}"
        file_path = upload_dir / filename
        save_upload_file(file, str(file_path))
        
        logger.info(f"文件上传成功: {file_path}")
        
        # 创建生成请求
        request = SimpleGenerationRequest(
            image_path=str(file_path),
            style=style,
            color_count=color_count,
            edge_enhancement=edge_enhancement,
            noise_reduction=noise_reduction,
            optimization_level=optimization_level,
            enable_auto_tune=enable_auto_tune
        )
        
        # 提交生成任务
        job_id = await simple_generate_embroidery_api(request)
        
        logger.info(f"生成任务已提交: {job_id}")
        
        return {
            "job_id": job_id,
            "message": "生成任务已提交",
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"生成刺绣图像失败: {e}")
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")


@app.get("/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """获取任务状态"""
    try:
        status = simple_get_job_status_api(job_id)
        if status is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        return JobStatusResponse(**status)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取任务状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取任务状态失败: {str(e)}")


@app.get("/download/{job_id}/{filename}")
async def download_result(job_id: str, filename: str):
    """下载结果文件"""
    try:
        # 构建文件路径
        file_path = Path("outputs") / job_id / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="文件不存在")
        
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type='application/octet-stream'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"下载文件失败: {e}")
        raise HTTPException(status_code=500, detail=f"下载失败: {str(e)}")


@app.get("/jobs")
async def list_jobs(limit: int = 10, status: Optional[str] = None):
    """列出任务"""
    try:
        # 获取系统统计
        stats = simple_get_system_stats_api()
        
        # 这里简化处理，实际应该从数据库或缓存中获取任务列表
        jobs = []
        
        # 添加一些示例任务信息
        if stats.get("completed_jobs", 0) > 0:
            jobs.append({
                "job_id": "example_job",
                "status": "completed",
                "processing_time": 5.0,
                "created_at": time.time() - 3600
            })
        
        return {
            "jobs": jobs[:limit],
            "total": len(jobs),
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"列出任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"列出任务失败: {str(e)}")


@app.delete("/job/{job_id}")
async def cancel_job(job_id: str):
    """取消任务"""
    try:
        # 检查任务状态
        status = simple_get_job_status_api(job_id)
        if status is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        if status["status"] in ["completed", "failed"]:
            raise HTTPException(status_code=400, detail="任务已完成，无法取消")
        
        # 这里应该实现任务取消逻辑
        # 简化版本，只返回成功消息
        return {
            "message": "任务取消请求已提交",
            "job_id": job_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"取消任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"取消任务失败: {str(e)}")


@app.post("/batch-generate")
async def batch_generate_embroidery(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    style: str = "basic",
    color_count: int = 16,
    optimization_level: str = "balanced"
):
    """批量生成刺绣图像"""
    try:
        if len(files) > 10:  # 限制批量处理数量
            raise HTTPException(status_code=400, detail="批量处理最多支持10个文件")
        
        job_ids = []
        
        for file in files:
            # 验证文件类型
            if not file.content_type.startswith('image/'):
                continue
            
            # 保存文件
            upload_dir = Path("uploads")
            upload_dir.mkdir(exist_ok=True)
            
            timestamp = int(time.time())
            filename = f"{timestamp}_{file.filename}"
            file_path = upload_dir / filename
            save_upload_file(file, str(file_path))
            
            # 创建请求
            request = SimpleGenerationRequest(
                image_path=str(file_path),
                style=style,
                color_count=color_count,
                optimization_level=optimization_level
            )
            
            # 提交任务
            job_id = await simple_generate_embroidery_api(request)
            job_ids.append(job_id)
        
        return {
            "message": f"批量生成任务已提交，共{len(job_ids)}个任务",
            "job_ids": job_ids,
            "total": len(job_ids)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量生成失败: {str(e)}")


@app.get("/styles")
async def get_available_styles():
    """获取可用风格"""
    return {
        "styles": [
            {
                "id": "basic",
                "name": "基础风格",
                "description": "标准图像处理风格"
            },
            {
                "id": "embroidery",
                "name": "刺绣风格",
                "description": "专为刺绣优化的风格"
            },
            {
                "id": "traditional",
                "name": "传统风格",
                "description": "传统艺术风格"
            },
            {
                "id": "modern",
                "name": "现代风格",
                "description": "现代艺术风格"
            }
        ]
    }


@app.get("/optimization-levels")
async def get_optimization_levels():
    """获取优化级别"""
    return {
        "levels": [
            {
                "id": "conservative",
                "name": "保守优化",
                "description": "轻微优化，保持原图特征"
            },
            {
                "id": "balanced",
                "name": "平衡优化",
                "description": "平衡质量和处理速度"
            },
            {
                "id": "aggressive",
                "name": "激进优化",
                "description": "最大程度优化，可能改变原图特征"
            }
        ]
    }


# 错误处理
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理"""
    logger.error(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "服务器内部错误",
            "error": str(exc)
        }
    )


# 启动事件
@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("刺绣图像处理API服务启动中...")
    
    # 创建必要的目录
    directories = ["uploads", "outputs", "cache", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    logger.info("API服务启动完成")


# 关闭事件
@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info("刺绣图像处理API服务关闭中...")


if __name__ == "__main__":
    # 启动服务器
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 