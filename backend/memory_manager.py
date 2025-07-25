"""
智能内存管理器 - 优化版本
提供智能内存池、内存使用分析、缓存策略优化和内存泄漏检测
"""

import numpy as np
import cv2
import logging
import time
import threading
import weakref
import gc
import psutil
import os
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from collections import OrderedDict, defaultdict
import json
from enum import Enum
import tracemalloc


class MemoryType(Enum):
    """内存类型"""
    IMAGE = "image"
    FEATURE = "feature"
    MODEL = "model"
    CACHE = "cache"
    TEMPORARY = "temporary"


@dataclass
class MemoryBlock:
    """内存块"""
    id: str
    size: int
    memory_type: MemoryType
    creation_time: float
    last_access_time: float
    access_count: int
    data: Any
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MemoryStats:
    """内存统计"""
    total_allocated: int
    total_used: int
    total_free: int
    peak_usage: int
    cache_hits: int
    cache_misses: int
    evictions: int
    memory_by_type: Dict[str, int]
    allocation_count: int
    deallocation_count: int


class MemoryPool:
    """智能内存池"""
    
    def __init__(self, max_size: int = 1024 * 1024 * 1024,  # 1GB
                 chunk_size: int = 1024 * 1024):  # 1MB
        self.max_size = max_size
        self.chunk_size = chunk_size
        self.allocated_chunks: Dict[int, bool] = {}
        self.free_chunks: List[int] = []
        self._lock = threading.Lock()
        
        # 初始化空闲块
        for i in range(0, max_size, chunk_size):
            self.free_chunks.append(i)
    
    def allocate(self, size: int) -> Optional[int]:
        """分配内存块"""
        with self._lock:
            if not self.free_chunks:
                return None
            
            # 找到足够大的块
            for i, chunk_start in enumerate(self.free_chunks):
                if chunk_start + size <= self.max_size:
                    chunk_end = chunk_start + size
                    
                    # 检查是否有足够的连续空间
                    if self._check_continuous_space(chunk_start, chunk_end):
                        # 分配内存
                        self.allocated_chunks[chunk_start] = True
                        self.free_chunks.pop(i)
                        return chunk_start
            
            return None
    
    def _check_continuous_space(self, start: int, end: int) -> bool:
        """检查连续空间是否可用"""
        for i in range(start, end, self.chunk_size):
            if i in self.allocated_chunks:
                return False
        return True
    
    def deallocate(self, address: int):
        """释放内存块"""
        with self._lock:
            if address in self.allocated_chunks:
                del self.allocated_chunks[address]
                self.free_chunks.append(address)
                self.free_chunks.sort()
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """获取使用统计"""
        with self._lock:
            allocated_size = len(self.allocated_chunks) * self.chunk_size
            free_size = len(self.free_chunks) * self.chunk_size
            
            return {
                "max_size": self.max_size,
                "allocated_size": allocated_size,
                "free_size": free_size,
                "usage_percent": (allocated_size / self.max_size) * 100,
                "allocated_chunks": len(self.allocated_chunks),
                "free_chunks": len(self.free_chunks)
            }


class CachePolicy(Enum):
    """缓存策略"""
    LRU = "lru"  # 最近最少使用
    LFU = "lfu"  # 最少使用
    FIFO = "fifo"  # 先进先出
    RANDOM = "random"  # 随机


class SmartCache:
    """智能缓存"""
    
    def __init__(self, max_size: int = 100 * 1024 * 1024,  # 100MB
                 policy: CachePolicy = CachePolicy.LRU):
        self.max_size = max_size
        self.policy = policy
        self.current_size = 0
        self.cache: OrderedDict = OrderedDict()
        self.access_counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
        
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_requests": 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        with self._lock:
            self.stats["total_requests"] += 1
            
            if key in self.cache:
                # 缓存命中
                self.stats["hits"] += 1
                item = self.cache[key]
                
                # 更新访问信息
                self.access_counts[key] += 1
                
                # 根据策略更新顺序
                if self.policy == CachePolicy.LRU:
                    self.cache.move_to_end(key)
                
                return item["value"] if isinstance(item, dict) and "value" in item else item
            else:
                # 缓存未命中
                self.stats["misses"] += 1
                return None
    
    def put(self, key: str, value: Any, size: int):
        """添加缓存项"""
        with self._lock:
            # 如果已存在，先移除
            if key in self.cache:
                old_size = self.cache[key]["size"]
                self.current_size -= old_size
                del self.cache[key]
            
            # 确保有足够空间
            while self.current_size + size > self.max_size:
                self._evict_item()
            
            # 添加新项
            self.cache[key] = {
                "value": value,
                "size": size,
                "timestamp": time.time()
            }
            self.current_size += size
            self.access_counts[key] = 0
    
    def _evict_item(self):
        """驱逐缓存项"""
        if not self.cache:
            return
        
        if self.policy == CachePolicy.LRU:
            # 移除最久未使用的项
            key = next(iter(self.cache))
        elif self.policy == CachePolicy.LFU:
            # 移除最少使用的项
            key = min(self.access_counts.keys(), key=lambda k: self.access_counts[k])
        elif self.policy == CachePolicy.FIFO:
            # 移除最先添加的项
            key = next(iter(self.cache))
        else:  # RANDOM
            # 随机移除
            key = list(self.cache.keys())[0]
        
        # 移除项
        item = self.cache.pop(key)
        self.current_size -= item["size"]
        if key in self.access_counts:
            del self.access_counts[key]
        
        self.stats["evictions"] += 1
    
    def clear(self):
        """清空缓存"""
        with self._lock:
            self.cache.clear()
            self.access_counts.clear()
            self.current_size = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self._lock:
            hit_rate = (self.stats["hits"] / self.stats["total_requests"] * 100 
                       if self.stats["total_requests"] > 0 else 0)
            
            return {
                **self.stats,
                "hit_rate": hit_rate,
                "current_size": self.current_size,
                "max_size": self.max_size,
                "usage_percent": (self.current_size / self.max_size) * 100,
                "item_count": len(self.cache)
            }


class MemoryManager:
    """
    智能内存管理器 - 优化版本
    
    提供智能内存池、缓存策略优化和内存泄漏检测
    """
    
    def __init__(self, max_memory: int = 2 * 1024 * 1024 * 1024,  # 2GB
                 enable_tracking: bool = True,
                 enable_gc: bool = True):
        self.max_memory = max_memory
        self.enable_tracking = enable_tracking
        self.enable_gc = enable_gc
        
        # 内存池
        self.memory_pool = MemoryPool(max_memory // 2)
        
        # 缓存
        self.image_cache = SmartCache(max_memory // 4, CachePolicy.LRU)
        self.feature_cache = SmartCache(max_memory // 8, CachePolicy.LFU)
        self.model_cache = SmartCache(max_memory // 8, CachePolicy.LRU)
        
        # 内存跟踪
        self.memory_blocks: Dict[str, MemoryBlock] = {}
        self.memory_stats = MemoryStats(
            total_allocated=0,
            total_used=0,
            total_free=0,
            peak_usage=0,
            cache_hits=0,
            cache_misses=0,
            evictions=0,
            memory_by_type=defaultdict(int),
            allocation_count=0,
            deallocation_count=0
        )
        
        # 内存泄漏检测
        self.leak_detector = MemoryLeakDetector() if enable_tracking else None
        
        # 线程安全
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # 启动内存监控
        if enable_tracking:
            self._start_memory_monitoring()
    
    def _start_memory_monitoring(self):
        """启动内存监控"""
        def monitor_memory():
            while True:
                try:
                    self._update_memory_stats()
                    time.sleep(5)  # 每5秒更新一次
                except Exception as e:
                    self.logger.error(f"内存监控错误: {e}")
        
        monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
        monitor_thread.start()
    
    def _update_memory_stats(self):
        """更新内存统计"""
        with self._lock:
            # 获取系统内存信息
            system_memory = psutil.virtual_memory()
            
            # 更新统计
            self.memory_stats.total_allocated = system_memory.total
            self.memory_stats.total_used = system_memory.used
            self.memory_stats.total_free = system_memory.available
            
            # 更新峰值使用
            if system_memory.used > self.memory_stats.peak_usage:
                self.memory_stats.peak_usage = system_memory.used
            
            # 更新缓存统计
            image_stats = self.image_cache.get_stats()
            feature_stats = self.feature_cache.get_stats()
            model_stats = self.model_cache.get_stats()
            
            self.memory_stats.cache_hits = (image_stats["hits"] + 
                                          feature_stats["hits"] + 
                                          model_stats["hits"])
            self.memory_stats.cache_misses = (image_stats["misses"] + 
                                            feature_stats["misses"] + 
                                            model_stats["misses"])
            self.memory_stats.evictions = (image_stats["evictions"] + 
                                         feature_stats["evictions"] + 
                                         model_stats["evictions"])
    
    def allocate_image(self, image: np.ndarray, 
                      image_id: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """分配图像内存"""
        if image_id is None:
            image_id = f"image_{int(time.time() * 1000)}"
        
        size = image.nbytes
        
        # 检查缓存
        cached_image = self.image_cache.get(image_id)
        if cached_image is not None:
            return image_id
        
        # 分配内存
        with self._lock:
            # 检查内存限制
            if self._check_memory_limit(size):
                # 添加到缓存
                self.image_cache.put(image_id, image, size)
                
                # 创建内存块
                memory_block = MemoryBlock(
                    id=image_id,
                    size=size,
                    memory_type=MemoryType.IMAGE,
                    creation_time=time.time(),
                    last_access_time=time.time(),
                    access_count=1,
                    data=image,
                    metadata=metadata
                )
                
                self.memory_blocks[image_id] = memory_block
                self.memory_stats.memory_by_type[MemoryType.IMAGE.value] += size
                self.memory_stats.allocation_count += 1
                
                # 内存泄漏检测
                if self.leak_detector:
                    self.leak_detector.track_allocation(image_id, size, MemoryType.IMAGE)
                
                return image_id
            else:
                raise MemoryError(f"内存不足，无法分配 {size} 字节")
    
    def get_image(self, image_id: str) -> Optional[np.ndarray]:
        """获取图像"""
        # 从缓存获取
        image = self.image_cache.get(image_id)
        
        if image is not None:
            # 更新访问信息
            with self._lock:
                if image_id in self.memory_blocks:
                    block = self.memory_blocks[image_id]
                    block.last_access_time = time.time()
                    block.access_count += 1
                    
                    if self.leak_detector:
                        self.leak_detector.track_access(image_id)
        
        return image
    
    def deallocate_image(self, image_id: str):
        """释放图像内存"""
        with self._lock:
            if image_id in self.memory_blocks:
                block = self.memory_blocks[image_id]
                
                # 从缓存移除
                self.image_cache.cache.pop(image_id, None)
                
                # 更新统计
                self.memory_stats.memory_by_type[MemoryType.IMAGE.value] -= block.size
                self.memory_stats.deallocation_count += 1
                
                # 移除内存块
                del self.memory_blocks[image_id]
                
                # 内存泄漏检测
                if self.leak_detector:
                    self.leak_detector.track_deallocation(image_id)
    
    def _check_memory_limit(self, size: int) -> bool:
        """检查内存限制"""
        current_usage = sum(block.size for block in self.memory_blocks.values())
        return current_usage + size <= self.max_memory
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计"""
        with self._lock:
            # 手动构建统计字典，避免defaultdict序列化问题
            stats = {
                "total_allocated": self.memory_stats.total_allocated,
                "total_used": self.memory_stats.total_used,
                "total_free": self.memory_stats.total_free,
                "peak_usage": self.memory_stats.peak_usage,
                "cache_hits": self.memory_stats.cache_hits,
                "cache_misses": self.memory_stats.cache_misses,
                "evictions": self.memory_stats.evictions,
                "memory_by_type": dict(self.memory_stats.memory_by_type),
                "allocation_count": self.memory_stats.allocation_count,
                "deallocation_count": self.memory_stats.deallocation_count
            }
            
            # 添加缓存统计
            stats["image_cache"] = self.image_cache.get_stats()
            stats["feature_cache"] = self.feature_cache.get_stats()
            stats["model_cache"] = self.model_cache.get_stats()
            
            # 添加内存池统计
            stats["memory_pool"] = self.memory_pool.get_usage_stats()
            
            # 添加泄漏检测统计
            if self.leak_detector:
                stats["leak_detection"] = self.leak_detector.get_stats()
            
            return stats
    
    def optimize_memory(self):
        """优化内存使用"""
        with self._lock:
            # 清理长时间未访问的内存块
            current_time = time.time()
            threshold = 300  # 5分钟
            
            to_remove = []
            for block_id, block in self.memory_blocks.items():
                if current_time - block.last_access_time > threshold:
                    to_remove.append(block_id)
            
            for block_id in to_remove:
                self.deallocate_image(block_id)
            
            # 强制垃圾回收
            if self.enable_gc:
                gc.collect()
            
            # 清理缓存
            if len(self.memory_blocks) > 100:  # 如果内存块过多
                self.image_cache.clear()
                self.feature_cache.clear()
                self.model_cache.clear()
    
    def export_memory_report(self, filepath: str):
        """导出内存报告"""
        try:
            report = {
                "timestamp": time.time(),
                "memory_stats": self.get_memory_stats(),
                "memory_blocks": {
                    block_id: asdict(block) for block_id, block in self.memory_blocks.items()
                }
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"内存报告已导出到: {filepath}")
            
        except Exception as e:
            self.logger.error(f"导出内存报告失败: {e}")


class MemoryLeakDetector:
    """内存泄漏检测器"""
    
    def __init__(self):
        self.allocations: Dict[str, Dict[str, Any]] = {}
        self.access_patterns: Dict[str, List[float]] = defaultdict(list)
        self.deallocations: set = set()
        
        # 启动跟踪
        tracemalloc.start()
    
    def track_allocation(self, block_id: str, size: int, memory_type: MemoryType):
        """跟踪内存分配"""
        self.allocations[block_id] = {
            "size": size,
            "type": memory_type.value,
            "timestamp": time.time(),
            "stack_trace": tracemalloc.get_object_traceback(block_id)
        }
    
    def track_access(self, block_id: str):
        """跟踪内存访问"""
        self.access_patterns[block_id].append(time.time())
    
    def track_deallocation(self, block_id: str):
        """跟踪内存释放"""
        self.deallocations.add(block_id)
        if block_id in self.allocations:
            del self.allocations[block_id]
    
    def detect_leaks(self) -> List[Dict[str, Any]]:
        """检测内存泄漏"""
        leaks = []
        
        for block_id, allocation in self.allocations.items():
            if block_id not in self.deallocations:
                # 检查访问模式
                access_times = self.access_patterns.get(block_id, [])
                last_access = max(access_times) if access_times else allocation["timestamp"]
                
                # 如果超过1小时未访问，可能是泄漏
                if time.time() - last_access > 3600:
                    leaks.append({
                        "block_id": block_id,
                        "size": allocation["size"],
                        "type": allocation["type"],
                        "allocation_time": allocation["timestamp"],
                        "last_access": last_access,
                        "access_count": len(access_times),
                        "stack_trace": allocation["stack_trace"]
                    })
        
        return leaks
    
    def get_stats(self) -> Dict[str, Any]:
        """获取泄漏检测统计"""
        leaks = self.detect_leaks()
        total_leaked_size = sum(leak["size"] for leak in leaks)
        
        return {
            "total_allocations": len(self.allocations),
            "total_deallocations": len(self.deallocations),
            "potential_leaks": len(leaks),
            "total_leaked_size": total_leaked_size,
            "leak_details": leaks
        }


# 全局内存管理器实例
_global_memory_manager = None

def get_global_memory_manager() -> MemoryManager:
    """获取全局内存管理器"""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager()
    return _global_memory_manager

def set_global_memory_manager(manager: MemoryManager):
    """设置全局内存管理器"""
    global _global_memory_manager
    _global_memory_manager = manager


# 测试函数
def test_memory_manager():
    """测试内存管理器"""
    manager = MemoryManager(max_memory=100 * 1024 * 1024)  # 100MB
    
    print("=== 内存管理器测试 ===")
    
    # 创建测试图像
    test_images = []
    for i in range(5):
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        image_id = f"test_image_{i}"
        
        try:
            allocated_id = manager.allocate_image(image, image_id)
            test_images.append(allocated_id)
            print(f"分配图像 {i}: {allocated_id}")
        except MemoryError as e:
            print(f"分配图像 {i} 失败: {e}")
            break
    
    # 测试获取图像
    for image_id in test_images:
        retrieved_image = manager.get_image(image_id)
        if retrieved_image is not None:
            print(f"获取图像成功: {image_id}, 形状: {retrieved_image.shape}")
    
    # 显示统计
    stats = manager.get_memory_stats()
    print(f"内存统计: {json.dumps(stats, indent=2, default=str)}")
    
    # 测试内存优化
    manager.optimize_memory()
    print("内存优化完成")
    
    # 清理
    for image_id in test_images:
        manager.deallocate_image(image_id)
    
    print("测试完成")


if __name__ == "__main__":
    test_memory_manager() 