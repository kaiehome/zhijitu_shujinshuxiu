import os
import sys
import argparse
import json
import csv
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from simple_professional_generator import SimpleProfessionalGenerator
from structural_professional_generator import StructuralProfessionalGenerator

# 日志配置
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "batch_infer.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_task_list(input_dir: str = None, task_csv: str = None) -> List[Dict[str, Any]]:
    """加载任务清单，支持目录遍历或CSV清单"""
    tasks = []
    if task_csv:
        with open(task_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                tasks.append({
                    'input_path': row['input_path'],
                    'job_id': row.get('job_id') or Path(row['input_path']).stem,
                    'params': json.loads(row.get('params', '{}'))
                })
    elif input_dir:
        for file in Path(input_dir).glob("*.jpg"):
            tasks.append({
                'input_path': str(file),
                'job_id': file.stem,
                'params': {}
            })
        for file in Path(input_dir).glob("*.png"):
            tasks.append({
                'input_path': str(file),
                'job_id': file.stem,
                'params': {}
            })
    else:
        logger.error("必须指定输入目录或任务清单CSV文件")
        sys.exit(1)
    return tasks

def process_single_task(task: Dict[str, Any], output_dir: str, generator_type: str = "simple") -> Dict[str, Any]:
    """处理单个推理任务"""
    try:
        input_path = task['input_path']
        job_id = task['job_id']
        params = task.get('params', {})
        os.makedirs(output_dir, exist_ok=True)
        if generator_type == "simple":
            generator = SimpleProfessionalGenerator()
            professional_path, comparison_path, processing_time = generator.generate_professional_image(
                input_path, job_id, **params
            )
        else:
            generator = StructuralProfessionalGenerator()
            professional_path, comparison_path, structure_info_path, processing_time = generator.generate_structural_professional_image(
                input_path, job_id, **params
            )
        result = {
            'job_id': job_id,
            'status': 'success',
            'output_files': {
                'professional': professional_path,
                'comparison': comparison_path
            },
            'processing_time': processing_time,
            'error_msg': ''
        }
        if generator_type != "simple":
            result['output_files']['structure_info'] = structure_info_path
        logger.info(f"任务 {job_id} 成功，耗时 {processing_time:.2f}s")
        return result
    except Exception as e:
        logger.error(f"任务 {task['job_id']} 失败: {str(e)}")
        return {
            'job_id': task['job_id'],
            'status': 'fail',
            'output_files': {},
            'processing_time': 0,
            'error_msg': str(e)
        }

def validate_output(result: Dict[str, Any]) -> bool:
    """校验输出文件是否存在"""
    if result['status'] != 'success':
        return False
    for f in result['output_files'].values():
        if not os.path.exists(f):
            logger.warning(f"输出文件不存在: {f}")
            return False
    return True

def save_report(report: List[Dict[str, Any]], report_path: str):
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"批量处理报告已保存: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="批量推理自动化脚本")
    parser.add_argument('--input_dir', type=str, help='输入图片目录')
    parser.add_argument('--task_csv', type=str, help='任务清单CSV文件')
    parser.add_argument('--output_dir', type=str, default='outputs/batch', help='输出目录')
    parser.add_argument('--generator', type=str, choices=['simple', 'structural'], default='simple', help='生成器类型')
    parser.add_argument('--num_workers', type=int, default=4, help='并发线程数')
    parser.add_argument('--report', type=str, default='outputs/batch/report.json', help='批量报告输出路径')
    args = parser.parse_args()

    tasks = load_task_list(args.input_dir, args.task_csv)
    results = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_task = {executor.submit(process_single_task, task, args.output_dir, args.generator): task for task in tasks}
        for future in as_completed(future_to_task):
            result = future.result()
            result['validated'] = validate_output(result)
            results.append(result)
    save_report(results, args.report)
    logger.info(f"批量任务完成，总数: {len(results)}，成功: {sum(1 for r in results if r['status']=='success')}，失败: {sum(1 for r in results if r['status']=='fail')}")

if __name__ == "__main__":
    main() 