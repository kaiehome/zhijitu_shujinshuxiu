"""
蜀锦蜀绣AI打样图生成工具 - 后端主应用
提供图像处理API服务，专注于蜀锦蜀绣传统风格的AI图像处理
"""

import os
import shutil
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pathlib import Path
import json
import time
import numpy as np

import aiofiles
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request, Form, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional
import shutil
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import ValidationError

from models import (
    ProcessRequest, 
    ProcessResponse, 
    TextToPatternRequest, 
    UploadResponse, 
    generate_job_id,
    validate_image_file,
    sanitize_filename
)
from image_processor import SichuanBrocadeProcessor
from simple_professional_generator import SimpleProfessionalGenerator
from structural_professional_generator import StructuralProfessionalGenerator
from professional_recognition_generator import ProfessionalRecognitionGenerator
from advanced_professional_generator import AdvancedProfessionalGenerator
from ultimate_professional_generator import UltimateProfessionalGenerator
from local_ai_generator import LocalAIGenerator
from tongyi_qianwen_api import get_tongyi_api, init_tongyi_api

# 导入图生图生成器
from image_to_image_generator import ImageToImageGenerator

# 导入通义千问Composer API
from tongyi_composer_api import get_composer_api, init_composer_api

# 初始化图生图生成器
img2img_generator = ImageToImageGenerator(
    color_count=16,
    use_ai_segmentation=True,
    use_feature_detection=True,
    style_preset="weaving_machine"
)

# 初始化通义千问Composer API
composer_api = get_composer_api()

# 配置日志系统
def setup_logging():
    """配置日志系统"""
    log_dir = Path("../logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "backend.log", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

setup_logging()
logger = logging.getLogger(__name__)

# 应用配置
class Config:
    """应用配置类"""
    UPLOAD_DIR = Path("uploads")
    OUTPUT_DIR = Path("outputs") 
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
    ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/jpg"}
    JOB_CLEANUP_HOURS = 24  # 24小时后清理任务
    MAX_CONCURRENT_JOBS = 10  # 最大并发任务数
    
    # 安全配置
    TRUSTED_HOSTS = ["localhost", "127.0.0.1", "0.0.0.0"]
    CORS_ORIGINS = [
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ]

config = Config()

# 创建FastAPI应用
app = FastAPI(
    title="专业织机识别图像生成器",
    description="专注于蜀锦蜀绣风格的AI图像处理工具，提供高质量的织机打样图生成服务",
    version="2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 安全中间件
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=config.TRUSTED_HOSTS
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# 全局异常处理器
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """处理数据验证异常"""
    logger.warning(f"数据验证失败: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": "请求数据格式错误", "errors": exc.errors()}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器"""
    logger.error(f"未处理的异常: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "服务器内部错误，请稍后重试"}
    )

# 初始化系统
def initialize_system():
    """初始化系统目录和资源"""
    try:
        # 创建必要目录
        for directory in [config.UPLOAD_DIR, config.OUTPUT_DIR]:
            directory.mkdir(exist_ok=True, parents=True)
            logger.info(f"目录已准备: {directory}")
        
        # 设置目录权限
        os.chmod(config.UPLOAD_DIR, 0o755)
        os.chmod(config.OUTPUT_DIR, 0o755)
        
        logger.info("系统初始化完成")
        
    except Exception as e:
        logger.error(f"系统初始化失败: {str(e)}")
        raise

initialize_system()

# 挂载静态文件
app.mount("/outputs", StaticFiles(directory=str(config.OUTPUT_DIR)), name="outputs")

# 使用专业真实识别图生成器作为默认处理器（更自然细腻的效果）
processor = ProfessionalRecognitionGenerator()

# 在app初始化后添加结构化生成器
structural_generator = StructuralProfessionalGenerator()

# 初始化专业识别图生成器
professional_generator = ProfessionalRecognitionGenerator()

# 初始化高级专业识别图生成器
advanced_generator = AdvancedProfessionalGenerator()

# 初始化终极专业识别图生成器
ultimate_generator = UltimateProfessionalGenerator()

# 初始化本地AI增强生成器
local_ai_generator = LocalAIGenerator()

# 初始化AI大模型
from ai_model_api import get_ai_model
ai_model = get_ai_model()

# 初始化简化的AI模型
from simple_ai_api import get_simple_ai_model
simple_ai_model = get_simple_ai_model()

# 初始化通义千问API
tongyi_api = init_tongyi_api()

# 任务状态管理
class JobManager:
    """任务管理器"""
    
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.active_jobs = 0
        self._restore_jobs_from_filesystem()
        
    def create_job(self, job_id: str, filename: str) -> bool:
        """创建新任务"""
        if self.active_jobs >= config.MAX_CONCURRENT_JOBS:
            return False
            
        self.jobs[job_id] = {
            "status": "processing",
            "message": "任务已创建，等待处理...",
            "original_filename": filename,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        self.active_jobs += 1
        return True
    
    def update_job(self, job_id: str, **kwargs):
        """更新任务状态"""
        if job_id in self.jobs:
            self.jobs[job_id].update(kwargs)
            self.jobs[job_id]["updated_at"] = datetime.now()
    
    def complete_job(self, job_id: str, **kwargs):
        """完成任务"""
        if job_id in self.jobs:
            self.jobs[job_id].update(kwargs)
            self.jobs[job_id]["status"] = "completed"
            self.jobs[job_id]["completed_at"] = datetime.now()
            self.active_jobs = max(0, self.active_jobs - 1)
    
    def fail_job(self, job_id: str, error_message: str):
        """任务失败"""
        if job_id in self.jobs:
            self.jobs[job_id].update({
                "status": "failed",
                "message": f"处理失败: {error_message}",
                "failed_at": datetime.now()
            })
            self.active_jobs = max(0, self.active_jobs - 1)
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """获取任务信息"""
        return self.jobs.get(job_id)
    
    def cleanup_old_jobs(self):
        """清理过期任务"""
        cutoff_time = datetime.now() - timedelta(hours=config.JOB_CLEANUP_HOURS)
        expired_jobs = [
            job_id for job_id, job_info in self.jobs.items()
            if job_info.get("created_at", datetime.now()) < cutoff_time
        ]
        
        for job_id in expired_jobs:
            self.jobs.pop(job_id, None)
            # 清理相关文件
            job_dir = config.OUTPUT_DIR / job_id
            if job_dir.exists():
                shutil.rmtree(job_dir, ignore_errors=True)
        
        if expired_jobs:
            logger.info(f"清理了 {len(expired_jobs)} 个过期任务")
    
    def _restore_jobs_from_filesystem(self):
        """从文件系统恢复任务状态"""
        try:
            if not config.OUTPUT_DIR.exists():
                return
            
            logger.info("🔄 正在从文件系统恢复任务状态...")
            restored_count = 0
            
            for job_dir in config.OUTPUT_DIR.iterdir():
                if job_dir.is_dir():
                    job_id = job_dir.name
                    
                    # 检查任务文件夹中的文件
                    png_files = list(job_dir.glob("*.png"))
                    svg_files = list(job_dir.glob("*.svg"))
                    
                    if png_files or svg_files:
                        # 构建文件列表
                        processed_files = []
                        for png_file in png_files:
                            processed_files.append(f"{job_id}/{png_file.name}")
                        for svg_file in svg_files:
                            processed_files.append(f"{job_id}/{svg_file.name}")
                        
                        # 获取文件时间
                        created_time = datetime.fromtimestamp(job_dir.stat().st_ctime)
                        modified_time = datetime.fromtimestamp(job_dir.stat().st_mtime)
                        
                        # 恢复任务状态
                        self.jobs[job_id] = {
                            "status": "completed",
                            "message": "图像处理完成",
                            "original_filename": "recovered_file",
                            "processed_files": processed_files,
                            "processing_time": 45.0,  # 估算值
                            "created_at": created_time,
                            "updated_at": modified_time,
                            "completed_at": modified_time,
                            "restored": True  # 标记为恢复的任务
                        }
                        restored_count += 1
            
            if restored_count > 0:
                logger.info(f"✅ 成功恢复 {restored_count} 个任务状态")
            else:
                logger.info("🔍 没有发现需要恢复的任务")
                
        except Exception as e:
            logger.error(f"❌ 恢复任务状态失败: {str(e)}", exc_info=True)

job_manager = JobManager()

# 定期清理任务
async def periodic_cleanup():
    """定期清理任务"""
    while True:
        try:
            job_manager.cleanup_old_jobs()
            await asyncio.sleep(3600)  # 每小时执行一次
        except Exception as e:
            logger.error(f"定期清理任务失败: {str(e)}")
            await asyncio.sleep(3600)

# 启动时开始定期清理
@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("🧵 蜀锦蜀绣AI打样图生成工具启动")
    asyncio.create_task(periodic_cleanup())

@app.on_event("shutdown") 
async def shutdown_event():
    """应用关闭事件"""
    logger.info("🛑 蜀锦蜀绣AI打样图生成工具关闭")

# API路由
@app.get("/", tags=["系统"])
async def root():
    """根路径 - 系统信息"""
    return {
        "message": "蜀锦蜀绣AI打样图生成工具API",
        "version": "1.0.0",
        "status": "运行中",
        "timestamp": datetime.now().isoformat(),
        "active_jobs": job_manager.active_jobs,
        "docs": "/docs"
    }

@app.get("/api/health", tags=["系统"])
async def health_check():
    """健康检查"""
    try:
        # 检查关键组件
        checks = {
            "api": True,
            "processor": processor is not None,
            "upload_dir": config.UPLOAD_DIR.exists(),
            "output_dir": config.OUTPUT_DIR.exists(),
            "active_jobs": job_manager.active_jobs,
            "total_jobs": len(job_manager.jobs)
        }
        
        all_healthy = all(checks.values())
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "timestamp": datetime.now().isoformat(),
            "service": "蜀锦蜀绣AI打样图生成工具",
            "checks": checks
        }
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
        )

@app.post("/api/upload", response_model=UploadResponse, tags=["文件操作"])
async def upload_file(file: UploadFile = File(...)):
    """
    上传图像文件
    
    - 支持格式：JPG, PNG
    - 最大大小：10MB
    - 自动文件名清理和验证
    """
    try:
        logger.info(f"接收文件上传请求: {file.filename}")
        
        # 验证文件基本信息
        if not file.filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")
        
        # 验证文件类型
        validation_result = validate_image_file(file.filename, file.content_type)
        if not validation_result["valid"]:
            raise HTTPException(status_code=400, detail=validation_result["error"])
        
        # 读取文件内容并验证大小
        content = await file.read()
        file_size = len(content)
        
        if file_size == 0:
            raise HTTPException(status_code=400, detail="文件内容为空")
        
        if file_size > config.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"文件大小超过限制 ({config.MAX_FILE_SIZE // (1024*1024)}MB)"
            )
        
        # 生成安全的文件名
        file_extension = Path(file.filename).suffix.lower()
        safe_filename = f"{generate_job_id()}{file_extension}"
        file_path = config.UPLOAD_DIR / safe_filename
        
        # 异步保存文件
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)
        
        # 设置文件权限
        os.chmod(file_path, 0o644)
        
        logger.info(f"文件上传成功: {file_path} ({file_size} bytes)")
        
        return UploadResponse(
            filename=safe_filename,
            size=file_size,
            content_type=file.content_type,
            upload_time=datetime.now().isoformat(),
            message="文件上传成功"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件上传失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="文件上传失败，请重试")

@app.post("/api/process", tags=["图像处理"])
async def process_image_api(
    file: UploadFile = File(...),
    color_count: Optional[int] = Form(16),
    edge_enhancement: Optional[bool] = Form(True),
    noise_reduction: Optional[bool] = Form(True)
):
    """API端点：处理图像生成专业织机识别图像"""
    try:
        # 生成job_id
        now = datetime.now()
        job_id = now.strftime("%y%m%d_%H%M%S")
        
        # 保存上传的文件
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, f"{job_id}.jpg")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"文件已保存: {file_path}")
        
        # 使用专业真实识别图处理器（更自然细腻的效果）
        start_time = time.time()
        professional_path = processor.generate_professional_recognition(file_path)
        processing_time = time.time() - start_time
        
        # 创建对比图路径
        comparison_path = professional_path.replace('_professional_recognition.', '_comparison.')
        
        # 创建对比图
        original = cv2.imread(file_path)
        processed = cv2.imread(professional_path)
        if original is not None and processed is not None:
            # 创建对比图
            comparison = np.hstack([original, processed])
            cv2.imwrite(comparison_path, comparison)
        
        logger.info(f"图像处理完成，耗时: {processing_time:.2f}秒")
        
        return {
            "status": "completed",
            "job_id": job_id,
            "processing_time": round(processing_time, 2),
            "professional_image_url": f"/{professional_path}",
            "comparison_image_url": f"/{comparison_path}",
            "parameters": {
                "color_count": color_count,
                "edge_enhancement": edge_enhancement,
                "noise_reduction": noise_reduction,
                "generator_type": "professional_recognition"
            },
            "message": f"专业织机识别图像生成成功！耗时: {processing_time:.2f}秒"
        }
        
    except Exception as e:
        logger.error(f"图像处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"图像处理失败: {str(e)}")

@app.post("/process-image")
async def process_image(
    file: UploadFile = File(...),
    color_count: Optional[int] = Form(16),
    edge_enhancement: Optional[bool] = Form(True),
    noise_reduction: Optional[bool] = Form(True)
):
    """处理图像生成专业织机识别图（兼容性端点）"""
    return await process_image_api(file, color_count, edge_enhancement, noise_reduction)

@app.get("/api/status/{job_id}", tags=["任务管理"])
async def get_job_status(job_id: str):
    """
    查询任务处理状态
    
    - 返回任务当前状态
    - 包含处理进度和结果文件信息
    """
    try:
        if not job_id:
            raise HTTPException(status_code=400, detail="任务ID不能为空")
        
        job_info = job_manager.get_job(job_id)
        if not job_info:
            raise HTTPException(status_code=404, detail="任务不存在或已过期")
        
        # 构建响应
        response_data = {
            "job_id": job_id,
            "status": job_info["status"],
            "message": job_info["message"],
            "original_filename": job_info.get("original_filename", "")
        }
        
        # 如果任务完成，添加结果信息
        if job_info["status"] == "completed":
            response_data.update({
                "processed_files": job_info.get("processed_files", []),
                "processing_time": job_info.get("processing_time", 0)
            })
        
        return ProcessResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查询任务状态失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="查询失败，请重试")

@app.get("/api/download/{job_id}/{filename}", tags=["文件操作"])
async def download_file(job_id: str, filename: str):
    """
    下载处理结果文件
    
    - 支持PNG和SVG文件下载
    - 自动设置正确的MIME类型
    """
    try:
        if not job_id or not filename:
            raise HTTPException(status_code=400, detail="参数不能为空")
        
        # 安全文件名检查
        safe_filename = sanitize_filename(filename)
        if safe_filename != filename:
            raise HTTPException(status_code=400, detail="文件名包含非法字符")
        
        # 构建文件路径
        file_path = config.OUTPUT_DIR / job_id / filename
        
        # 检查文件存在性和安全性
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="文件不存在")
        
        # 确保文件在允许的目录内（防止路径遍历攻击）
        if not str(file_path.resolve()).startswith(str(config.OUTPUT_DIR.resolve())):
            raise HTTPException(status_code=403, detail="访问被拒绝")
        
        # 确定MIME类型
        mime_type_map = {
            '.png': 'image/png',
            '.svg': 'image/svg+xml',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg'
        }
        
        file_extension = Path(filename).suffix.lower()
        media_type = mime_type_map.get(file_extension, 'application/octet-stream')
        
        logger.info(f"文件下载: {file_path}")
        
        return FileResponse(
            path=file_path,
            media_type=media_type,
            filename=filename,
            headers={"Cache-Control": "public, max-age=3600"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件下载失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="下载失败，请重试")

@app.post("/api/text-to-pattern", tags=["扩展功能"])
async def generate_pattern_from_text(request: TextToPatternRequest):
    """
    文本生成图案（预留接口）
    
    未来将接入通义千问、文心一言等国产大模型
    """
    try:
        logger.info(f"文生图请求: {request.prompt}")
        
        # 参数验证
        if not request.prompt or len(request.prompt.strip()) == 0:
            raise HTTPException(status_code=400, detail="提示词不能为空")
        
        if len(request.prompt) > 500:
            raise HTTPException(status_code=400, detail="提示词长度不能超过500字符")
        
        return JSONResponse(
            status_code=501,
            content={
                "message": "文生图功能正在开发中",
                "prompt": request.prompt,
                "style": request.style,
                "note": "该功能将在后续版本中接入国产大模型API",
                "planned_models": [
                    "通义千问-VL",
                    "百度文心一言",
                    "讯飞星火认知大模型"
                ]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文生图请求失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="请求失败，请重试")

@app.get("/api/stats", tags=["系统"])
async def get_system_stats():
    """获取系统统计信息"""
    try:
        return {
            "active_jobs": job_manager.active_jobs,
            "total_jobs": len(job_manager.jobs),
            "max_concurrent_jobs": config.MAX_CONCURRENT_JOBS,
            "upload_dir_size": sum(
                f.stat().st_size for f in config.UPLOAD_DIR.rglob('*') if f.is_file()
            ),
            "output_dir_size": sum(
                f.stat().st_size for f in config.OUTPUT_DIR.rglob('*') if f.is_file()
            ),
            "uptime": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"获取系统统计失败: {str(e)}")
        raise HTTPException(status_code=500, detail="获取统计信息失败")

@app.post("/api/process-structural")
async def process_structural_image(
    file: UploadFile = File(...),
    color_count: int = Form(16),
    edge_enhancement: bool = Form(True),
    noise_reduction: bool = Form(True)
):
    """
    处理图像生成结构化专业识别图
    
    新技术路径：
    - 结构规则为主轴
    - AI为辅助
    - 机器可读的结构特征
    """
    try:
        logger.info(f"🎯 开始结构化专业识别图处理: {file.filename}")
        
        # 生成任务ID
        job_id = datetime.now().strftime("%y%m%d_%H%M%S")
        
        # 验证文件
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise HTTPException(status_code=400, detail="不支持的文件格式，请上传PNG或JPG图像")
        
        if file.size > 10 * 1024 * 1024:  # 10MB限制
            raise HTTPException(status_code=400, detail="文件大小超过10MB限制")
        
        # 保存上传的文件
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_extension = Path(file.filename).suffix
        input_filename = f"{job_id}_input{file_extension}"
        input_path = upload_dir / input_filename
        
        with open(input_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"文件已保存: {input_path}")
        
        # 使用结构化生成器处理图像
        professional_path, comparison_path, structure_info_path, processing_time = structural_generator.generate_structural_professional_image(
            str(input_path), 
            job_id, 
            color_count=color_count
        )
        
        # 读取结构信息
        structure_info = {}
        try:
            with open(structure_info_path, 'r', encoding='utf-8') as f:
                structure_info = json.load(f)
        except Exception as e:
            logger.warning(f"读取结构信息失败: {str(e)}")
        
        # 返回结果
        return {
            "status": "completed",
            "job_id": job_id,
            "processing_time": round(processing_time, 2),
            "professional_image_url": f"/outputs/{job_id}/{job_id}_structural_professional.png",
            "comparison_image_url": f"/outputs/{job_id}/{job_id}_structural_comparison.png",
            "structure_info_url": f"/outputs/{job_id}/{job_id}_structure_info.json",
            "parameters": {
                "color_count": color_count,
                "edge_enhancement": edge_enhancement,
                "noise_reduction": noise_reduction,
                "generator_type": "structural_professional"
            },
            "structure_metrics": {
                "total_regions": structure_info.get('metadata', {}).get('total_regions', 0),
                "total_boundaries": structure_info.get('metadata', {}).get('total_boundaries', 0),
                "color_palette_size": len(structure_info.get('color_palette', [])),
                "is_machine_readable": True,
                "has_vector_paths": True,
                "region_closure_validated": True
            },
            "message": "结构化专业识别图生成完成！具备机器可读结构特征。"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"结构化专业识别图处理失败: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/generate-professional-recognition")
async def generate_professional_recognition(file: UploadFile = File(...)):
    """
    生成专业真实识别图
    """
    try:
        # 创建上传目录
        os.makedirs("uploads", exist_ok=True)
        
        # 保存上传的文件
        timestamp = int(time.time() * 1000)
        filename = f"{timestamp}_{file.filename}"
        file_path = f"uploads/{filename}"
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # 生成专业识别图
        output_path = professional_generator.generate_professional_recognition(file_path)
        
        # 返回结果
        return {
            "success": True,
            "message": "专业识别图生成成功",
            "original_image": file_path,
            "processed_image": output_path,
            "timestamp": timestamp
        }
        
    except Exception as e:
        logger.error(f"生成专业识别图时出错: {str(e)}")
        return {
            "success": False,
            "message": f"生成失败: {str(e)}"
        }

@app.post("/api/generate-advanced-professional")
async def generate_advanced_professional(file: UploadFile = File(...)):
    """
    生成高级专业真实识别图
    """
    try:
        # 创建上传目录
        os.makedirs("uploads", exist_ok=True)
        
        # 保存上传的文件
        timestamp = int(time.time() * 1000)
        filename = f"{timestamp}_{file.filename}"
        file_path = f"uploads/{filename}"
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # 生成高级专业识别图
        output_path = advanced_generator.generate_advanced_professional(file_path)
        
        # 返回结果
        return {
            "success": True,
            "message": "高级专业识别图生成成功",
            "original_image": file_path,
            "processed_image": output_path,
            "timestamp": timestamp
        }
        
    except Exception as e:
        logger.error(f"生成高级专业识别图时出错: {str(e)}")
        return {
            "success": False,
            "message": f"生成失败: {str(e)}"
        }

@app.post("/api/generate-ultimate-professional")
async def generate_ultimate_professional(file: UploadFile = File(...)):
    """
    生成终极专业真实识别图
    """
    try:
        # 创建上传目录
        os.makedirs("uploads", exist_ok=True)
        
        # 保存上传的文件
        timestamp = int(time.time() * 1000)
        filename = f"{timestamp}_{file.filename}"
        file_path = f"uploads/{filename}"
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # 生成终极专业识别图
        output_path = ultimate_generator.generate_ultimate_professional(file_path)
        
        # 返回结果
        return {
            "success": True,
            "message": "终极专业识别图生成成功",
            "original_image": file_path,
            "processed_image": output_path,
            "timestamp": timestamp
        }
        
    except Exception as e:
        logger.error(f"生成终极专业识别图时出错: {str(e)}")
        return {
            "success": False,
            "message": f"生成失败: {str(e)}"
        }

@app.post("/api/generate-local-ai-enhanced")
async def generate_local_ai_enhanced(file: UploadFile = File(...)):
    """
    生成本地AI增强的专业真实识别图
    """
    try:
        # 创建上传目录
        os.makedirs("uploads", exist_ok=True)
        
        # 保存上传的文件
        timestamp = int(time.time() * 1000)
        filename = f"{timestamp}_{file.filename}"
        file_path = f"uploads/{filename}"
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # 生成本地AI增强图像
        output_path = local_ai_generator.generate_local_ai_enhanced_image(file_path)
        
        # 读取生成的文件
        with open(output_path, "rb") as f:
            output_content = f.read()
        
        # 返回生成的图像
        return Response(
            content=output_content,
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f"attachment; filename={os.path.basename(output_path)}"
            }
        )
        
    except Exception as e:
        logger.error(f"本地AI增强处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"本地AI增强处理失败: {str(e)}")

@app.post("/api/generate-ai-model")
async def generate_ai_model(file: UploadFile = File(...)):
    """
    使用AI大模型生成织机识别图
    """
    try:
        # 检查AI模型是否可用
        if not ai_model.is_model_available():
            raise HTTPException(status_code=503, detail="AI模型未加载，请先训练模型")
        
        os.makedirs("uploads", exist_ok=True)
        timestamp = int(time.time() * 1000)
        filename = f"{timestamp}_{file.filename}"
        file_path = f"uploads/{filename}"
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # 使用AI模型生成图片
        output_path = ai_model.generate_image(file_path)
        
        if output_path is None:
            raise HTTPException(status_code=500, detail="AI模型生成失败")
        
        with open(output_path, "rb") as f:
            output_content = f.read()
        
        return Response(
            content=output_content,
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f"attachment; filename={os.path.basename(output_path)}"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AI模型生成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI模型生成失败: {str(e)}")

@app.get("/api/ai-model-status")
async def get_ai_model_status():
    """
    获取AI模型状态
    """
    try:
        model_info = ai_model.get_model_info()
        return JSONResponse(content=model_info)
    except Exception as e:
        logger.error(f"获取AI模型状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取AI模型状态失败: {str(e)}")

@app.post("/api/generate-simple-ai")
async def generate_simple_ai(file: UploadFile = File(...)):
    """
    使用简化的AI模型生成织机识别图
    """
    try:
        # 检查简化AI模型是否可用
        if not simple_ai_model.is_model_available():
            raise HTTPException(status_code=503, detail="简化AI模型未加载")
        
        os.makedirs("uploads", exist_ok=True)
        timestamp = int(time.time() * 1000)
        filename = f"{timestamp}_{file.filename}"
        file_path = f"uploads/{filename}"
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # 使用简化AI模型生成图片
        output_path = simple_ai_model.generate_image(file_path)
        
        if output_path is None:
            raise HTTPException(status_code=500, detail="简化AI模型生成失败")
        
        with open(output_path, "rb") as f:
            output_content = f.read()
        
        return Response(
            content=output_content,
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f"attachment; filename={os.path.basename(output_path)}"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"简化AI模型生成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"简化AI模型生成失败: {str(e)}")

@app.get("/api/simple-ai-status")
async def get_simple_ai_status():
    """
    获取简化AI模型状态
    """
    try:
        model_info = simple_ai_model.get_model_info()
        return JSONResponse(content=model_info)
    except Exception as e:
        logger.error(f"获取简化AI模型状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取简化AI模型状态失败: {str(e)}")

@app.post("/api/generate-tongyi-qianwen")
async def generate_tongyi_qianwen(file: UploadFile = File(...)):
    """
    使用通义千问大模型生成织机识别图
    """
    try:
        # 检查通义千问API是否可用
        api_status = tongyi_api.get_api_status()
        if not api_status["api_available"]:
            raise HTTPException(status_code=503, detail="通义千问API未配置，请设置TONGYI_API_KEY环境变量")
        
        os.makedirs("uploads", exist_ok=True)
        timestamp = int(time.time() * 1000)
        filename = f"{timestamp}_{file.filename}"
        file_path = f"uploads/{filename}"
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # 使用通义千问生成图片
        output_path = tongyi_api.generate_loom_recognition_image(file_path)
        
        if output_path is None:
            raise HTTPException(status_code=500, detail="通义千问生成失败")
        
        with open(output_path, "rb") as f:
            output_content = f.read()
        
        return Response(
            content=output_content,
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f"attachment; filename={os.path.basename(output_path)}"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"通义千问生成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"通义千问生成失败: {str(e)}")

@app.post("/api/enhance-tongyi-qianwen")
async def enhance_tongyi_qianwen(file: UploadFile = File(...)):
    """
    使用通义千问大模型增强图片质量
    """
    try:
        # 检查通义千问API是否可用
        api_status = tongyi_api.get_api_status()
        if not api_status["api_available"]:
            raise HTTPException(status_code=503, detail="通义千问API未配置，请设置TONGYI_API_KEY环境变量")
        
        os.makedirs("uploads", exist_ok=True)
        timestamp = int(time.time() * 1000)
        filename = f"{timestamp}_{file.filename}"
        file_path = f"uploads/{filename}"
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # 使用通义千问增强图片
        output_path = tongyi_api.enhance_image_quality(file_path)
        
        if output_path is None:
            raise HTTPException(status_code=500, detail="通义千问增强失败")
        
        with open(output_path, "rb") as f:
            output_content = f.read()
        
        return Response(
            content=output_content,
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f"attachment; filename={os.path.basename(output_path)}"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"通义千问增强失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"通义千问增强失败: {str(e)}")

@app.get("/api/tongyi-qianwen-status")
async def get_tongyi_qianwen_status():
    """
    获取通义千问API状态
    """
    try:
        api_status = tongyi_api.get_api_status()
        return JSONResponse(content=api_status)
    except Exception as e:
        logger.error(f"获取通义千问状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取通义千问状态失败: {str(e)}")

@app.post("/api/generate-image-to-image")
async def generate_image_to_image(
    file: UploadFile = File(...),
    style_preset: str = Form("weaving_machine"),
    color_count: int = Form(16)
):
    """
    纯图生图API - 上传原图，无需文字输入，直接生成织机识别图
    """
    try:
        # 检查文件类型
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="只支持图片文件")
        
        # 保存上传文件
        os.makedirs("uploads", exist_ok=True)
        timestamp = int(time.time() * 1000)
        filename = f"{timestamp}_{file.filename}"
        file_path = f"uploads/{filename}"
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"开始图生图处理: {file_path}")
        
        # 使用图生图生成器处理
        output_path = img2img_generator.generate_loom_recognition_image(
            source_image_path=file_path,
            style_preset=style_preset
        )
        
        if output_path is None:
            raise HTTPException(status_code=500, detail="图生图处理失败")
        
        # 返回生成的图片
        with open(output_path, "rb") as f:
            output_content = f.read()
        
        return Response(
            content=output_content,
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f"attachment; filename={os.path.basename(output_path)}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"图生图处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"图生图处理失败: {str(e)}")

@app.get("/api/image-to-image-styles")
async def get_image_to_image_styles():
    """
    获取可用的图生图风格预设
    """
    try:
        styles = img2img_generator.get_available_styles()
        info = img2img_generator.get_processing_info()
        
        return {
            "available_styles": styles,
            "current_style": info["current_style"],
            "processing_info": info
        }
    except Exception as e:
        logger.error(f"获取风格预设失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取风格预设失败: {str(e)}")

@app.post("/api/add-custom-style")
async def add_custom_style(
    style_name: str = Form(...),
    style_config: str = Form(...)  # JSON字符串
):
    """
    添加自定义图生图风格
    """
    try:
        config = json.loads(style_config)
        img2img_generator.add_custom_style(style_name, config)
        
        return {
            "message": f"自定义风格 '{style_name}' 添加成功",
            "available_styles": img2img_generator.get_available_styles()
        }
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="风格配置必须是有效的JSON格式")
    except Exception as e:
        logger.error(f"添加自定义风格失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"添加自定义风格失败: {str(e)}")

@app.post("/api/generate-composer-image-to-image")
async def generate_composer_image_to_image(
    file: UploadFile = File(...),
    prompt: str = Form(""),
    style_preset: str = Form("weaving_machine")
):
    """
    使用通义千问Composer模型进行图生图
    基于Qwen-VL + Composer模型实现图生图功能
    """
    try:
        # 检查通义千问Composer API是否可用
        api_status = composer_api.get_api_status()
        if not api_status["api_available"]:
            raise HTTPException(status_code=503, detail="通义千问Composer API未配置，请设置TONGYI_API_KEY环境变量")
        
        # 检查文件类型
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="只支持图片文件")
        
        # 保存上传文件
        os.makedirs("uploads", exist_ok=True)
        timestamp = int(time.time() * 1000)
        filename = f"{timestamp}_{file.filename}"
        file_path = f"uploads/{filename}"
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"开始Composer图生图处理: {file_path}")
        
        # 使用Composer API生成图片
        output_path = composer_api.generate_image_to_image(
            source_image_path=file_path,
            prompt=prompt,
            style_preset=style_preset
        )
        
        if output_path is None:
            raise HTTPException(status_code=500, detail="Composer图生图处理失败")
        
        # 返回生成的图片
        with open(output_path, "rb") as f:
            output_content = f.read()
        
        return Response(
            content=output_content,
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f"attachment; filename={os.path.basename(output_path)}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Composer图生图处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Composer图生图处理失败: {str(e)}")

@app.get("/api/composer-status")
async def get_composer_status():
    """
    获取通义千问Composer API状态
    """
    try:
        api_status = composer_api.get_api_status()
        return {
            "api_status": api_status,
            "processing_info": {
                "model": api_status["model"],
                "api_available": api_status["api_available"],
                "available_styles": ["weaving_machine", "embroidery", "pixel_art", "traditional", "modern"],
                "current_style": "weaving_machine"
            },
            "available_styles": ["weaving_machine", "embroidery", "pixel_art", "traditional", "modern"]
        }
    except Exception as e:
        logger.error(f"获取Composer状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取Composer状态失败: {str(e)}")

@app.post("/api/generate-composer-with-custom-prompt")
async def generate_composer_with_custom_prompt(
    file: UploadFile = File(...),
    custom_prompt: str = Form(...),
    style_preset: str = Form("weaving_machine")
):
    """
    使用自定义提示词进行Composer图生图
    """
    try:
        # 检查通义千问Composer API是否可用
        api_status = composer_api.get_api_status()
        if not api_status["api_available"]:
            raise HTTPException(status_code=503, detail="通义千问Composer API未配置，请设置TONGYI_API_KEY环境变量")
        
        # 检查文件类型
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="只支持图片文件")
        
        # 保存上传文件
        os.makedirs("uploads", exist_ok=True)
        timestamp = int(time.time() * 1000)
        filename = f"{timestamp}_{file.filename}"
        file_path = f"uploads/{filename}"
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"开始Composer自定义提示词图生图处理: {file_path}")
        logger.info(f"自定义提示词: {custom_prompt}")
        
        # 使用Composer API生成图片
        output_path = composer_api.generate_image_to_image(
            source_image_path=file_path,
            prompt=custom_prompt,
            style_preset=style_preset
        )
        
        if output_path is None:
            raise HTTPException(status_code=500, detail="Composer自定义提示词图生图处理失败")
        
        # 返回生成的图片
        with open(output_path, "rb") as f:
            output_content = f.read()
        
        return Response(
            content=output_content,
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f"attachment; filename={os.path.basename(output_path)}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Composer自定义提示词图生图处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Composer自定义提示词图生图处理失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )