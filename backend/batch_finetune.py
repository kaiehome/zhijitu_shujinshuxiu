import os
import sys
import argparse
import json
import csv
import logging
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
import numpy as np

# 日志配置
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "batch_finetune.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_task_list(task_csv: str) -> List[Dict[str, Any]]:
    """加载原图-识别图对任务清单（CSV）"""
    tasks = []
    with open(task_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tasks.append({
                'input_path': row['input_path'],
                'target_path': row['target_path'],
                'job_id': row.get('job_id') or Path(row['input_path']).stem,
                'params': json.loads(row.get('params', '{}'))
            })
    return tasks

def preprocess_image(img_path: str, size: int = 512) -> np.ndarray:
    """加载并预处理图片（resize, 归一化）"""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    return arr

def dummy_train_step(input_img: np.ndarray, target_img: np.ndarray, **kwargs) -> float:
    """模拟训练步骤，返回伪损失（实际应集成LoRA/Diffusers等训练模块）"""
    # 这里只做简单L2损失演示
    loss = np.mean((input_img - target_img) ** 2)
    return float(loss)

def validate_output(loss: float, threshold: float = 0.05) -> bool:
    """简单校验，损失低于阈值视为成功"""
    return loss < threshold

def save_report(report: List[Dict[str, Any]], report_path: str):
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"批量微调报告已保存: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="批量微调自动化脚本（原图-识别图对）")
    parser.add_argument('--task_csv', type=str, required=True, help='任务清单CSV文件')
    parser.add_argument('--img_size', type=int, default=512, help='图片resize尺寸')
    parser.add_argument('--report', type=str, default='outputs/batch_finetune_report.json', help='报告输出路径')
    args = parser.parse_args()

    tasks = load_task_list(args.task_csv)
    results = []
    for task in tasks:
        try:
            input_img = preprocess_image(task['input_path'], args.img_size)
            target_img = preprocess_image(task['target_path'], args.img_size)
            loss = dummy_train_step(input_img, target_img, **task['params'])
            status = 'success' if validate_output(loss) else 'fail'
            logger.info(f"任务 {task['job_id']} 损失: {loss:.4f} 状态: {status}")
            results.append({
                'job_id': task['job_id'],
                'input_path': task['input_path'],
                'target_path': task['target_path'],
                'loss': loss,
                'status': status
            })
        except Exception as e:
            logger.error(f"任务 {task['job_id']} 失败: {str(e)}")
            results.append({
                'job_id': task['job_id'],
                'input_path': task['input_path'],
                'target_path': task['target_path'],
                'loss': None,
                'status': 'fail',
                'error_msg': str(e)
            })
    save_report(results, args.report)
    logger.info(f"批量微调任务完成，总数: {len(results)}，成功: {sum(1 for r in results if r['status']=='success')}，失败: {sum(1 for r in results if r['status']=='fail')}")

if __name__ == "__main__":
    main() 