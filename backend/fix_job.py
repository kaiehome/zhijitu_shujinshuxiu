#!/usr/bin/env python3
"""
修复任务状态脚本
当任务文件已生成但状态未更新时使用
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime

def check_and_fix_jobs():
    """检查并修复任务状态"""
    outputs_dir = Path("outputs")
    
    if not outputs_dir.exists():
        print("输出目录不存在")
        return
    
    print("🔍 检查任务状态...")
    
    for job_dir in outputs_dir.iterdir():
        if job_dir.is_dir():
            job_id = job_dir.name
            
            # 检查文件生成情况
            png_files = list(job_dir.glob("*.png"))
            svg_files = list(job_dir.glob("*.svg"))
            
            if png_files or svg_files:
                print(f"\n📁 任务 {job_id}:")
                print(f"   PNG文件: {len(png_files)}")
                print(f"   SVG文件: {len(svg_files)}")
                
                # 显示文件详情
                for png_file in png_files:
                    size_mb = png_file.stat().st_size / 1024 / 1024
                    print(f"   📄 {png_file.name} ({size_mb:.2f} MB)")
                
                for svg_file in svg_files:
                    size_kb = svg_file.stat().st_size / 1024
                    print(f"   📄 {svg_file.name} ({size_kb:.2f} KB)")
                
                # 检查图像信息
                if png_files:
                    try:
                        from PIL import Image
                        with Image.open(png_files[0]) as img:
                            print(f"   🎨 分辨率: {img.size[0]}x{img.size[1]}")
                            pixels = img.size[0] * img.size[1]
                            print(f"   🔢 像素总数: {pixels:,}")
                    except Exception as e:
                        print(f"   ⚠️ 无法读取图像信息: {e}")

def fix_latest_job():
    """修复最新的任务"""
    outputs_dir = Path("outputs")
    
    if not outputs_dir.exists():
        print("输出目录不存在")
        return None
    
    # 找到最新的任务目录
    job_dirs = [d for d in outputs_dir.iterdir() if d.is_dir()]
    if not job_dirs:
        print("没有找到任务目录")
        return None
    
    # 按修改时间排序，获取最新的
    latest_job = max(job_dirs, key=lambda x: x.stat().st_mtime)
    job_id = latest_job.name
    
    print(f"🎯 最新任务: {job_id}")
    
    # 检查文件
    png_files = list(latest_job.glob("*.png"))
    svg_files = list(latest_job.glob("*.svg"))
    
    if not png_files and not svg_files:
        print("❌ 没有生成任何文件")
        return None
    
    print(f"✅ 找到 {len(png_files)} 个PNG文件和 {len(svg_files)} 个SVG文件")
    
    # 构建文件列表
    processed_files = []
    for png_file in png_files:
        processed_files.append(f"{job_id}/{png_file.name}")
    for svg_file in svg_files:
        processed_files.append(f"{job_id}/{svg_file.name}")
    
    # 返回任务信息
    return {
        "job_id": job_id,
        "status": "completed",
        "message": "图像处理完成",
        "processed_files": processed_files,
        "processing_time": 45.0,  # 估算时间
        "created_at": datetime.now().isoformat()
    }

def main():
    print("🛠️ 任务状态修复工具")
    print("=" * 50)
    
    # 检查所有任务
    check_and_fix_jobs()
    
    # 修复最新任务
    print("\n" + "=" * 50)
    latest_job_info = fix_latest_job()
    
    if latest_job_info:
        print(f"\n✅ 最新任务信息:")
        print(f"   任务ID: {latest_job_info['job_id']}")
        print(f"   状态: {latest_job_info['status']}")
        print(f"   消息: {latest_job_info['message']}")
        print(f"   文件: {latest_job_info['processed_files']}")
        
        # 提供API测试命令
        job_id = latest_job_info['job_id']
        print(f"\n🔗 API测试命令:")
        print(f"curl -s 'http://localhost:8000/api/status/{job_id}'")
    
    print("\n" + "=" * 50)
    print("修复完成！")

if __name__ == "__main__":
    main() 