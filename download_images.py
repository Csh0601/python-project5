# -*- coding: utf-8 -*-
"""
大规模图片数据集下载脚本
支持断点续传、多线程下载、错误处理和进度保存
"""

import os
import csv
import time
import hashlib
import requests
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
from typing import Optional, Set
import json
from datetime import datetime

# ===================== 配置参数 =====================
CONFIG = {
    # CSV文件路径
    "csv_file": "data.csv",
    
    # 图片保存目录
    "output_dir": "downloaded_images",
    
    # 进度文件（记录已下载的URL）
    "progress_file": "download_progress.json",
    
    # 失败记录文件
    "failed_file": "failed_downloads.txt",
    
    # 并发下载线程数（根据网络情况调整，建议10-50）
    "max_workers": 20,
    
    # 单个文件下载超时时间（秒）
    "timeout": 30,
    
    # 重试次数
    "max_retries": 3,
    
    # 重试延迟（秒）
    "retry_delay": 2,
    
    # 每批处理的行数（用于分批读取大CSV）
    "batch_size": 10000,
    
    # 是否保存caption到文本文件
    "save_captions": True,
    
    # 日志级别
    "log_level": logging.INFO,
    
    # 进度保存间隔（每下载多少张保存一次进度）
    "save_interval": 100,
}

# ===================== 日志配置 =====================
logging.basicConfig(
    level=CONFIG["log_level"],
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("download.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ImageDownloader:
    """图片下载器类"""
    
    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载已完成的下载记录
        self.completed: Set[str] = self._load_progress()
        self.failed_urls: list = []
        self.download_count = 0
        self.start_time = None
        
        # 请求session（复用连接）
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
    
    def _load_progress(self) -> Set[str]:
        """加载下载进度"""
        progress_file = Path(self.config["progress_file"])
        if progress_file.exists():
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.info(f"已加载进度，之前已下载 {len(data['completed'])} 个文件")
                    return set(data['completed'])
            except Exception as e:
                logger.warning(f"加载进度文件失败: {e}")
        return set()
    
    def _save_progress(self):
        """保存下载进度"""
        progress_file = Path(self.config["progress_file"])
        try:
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'completed': list(self.completed),
                    'last_update': datetime.now().isoformat(),
                    'total_downloaded': len(self.completed)
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存进度失败: {e}")
    
    def _get_filename_from_url(self, url: str, index: int) -> str:
        """从URL生成文件名"""
        try:
            parsed = urlparse(url)
            original_name = os.path.basename(parsed.path)
            
            # 提取扩展名
            ext = os.path.splitext(original_name)[1].lower()
            if ext not in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']:
                ext = '.jpg'  # 默认扩展名
            
            # 使用URL的hash作为唯一标识
            url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
            
            return f"{index:08d}_{url_hash}{ext}"
        except Exception:
            return f"{index:08d}.jpg"
    
    def _download_single(self, url: str, index: int, caption: str = "") -> Optional[str]:
        """下载单个图片"""
        if url in self.completed:
            return None  # 已下载，跳过
        
        filename = self._get_filename_from_url(url, index)
        filepath = self.output_dir / filename
        
        for attempt in range(self.config["max_retries"]):
            try:
                response = self.session.get(
                    url, 
                    timeout=self.config["timeout"],
                    stream=True
                )
                response.raise_for_status()
                
                # 检查是否是图片
                content_type = response.headers.get('Content-Type', '')
                if 'image' not in content_type and 'octet-stream' not in content_type:
                    logger.warning(f"非图片内容: {url} (Content-Type: {content_type})")
                    # 仍然保存，可能是有效图片
                
                # 写入文件
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # 保存caption（如果启用）
                if self.config["save_captions"] and caption:
                    caption_file = filepath.with_suffix('.txt')
                    with open(caption_file, 'w', encoding='utf-8') as f:
                        f.write(caption)
                
                return url  # 返回成功下载的URL
                
            except requests.exceptions.RequestException as e:
                if attempt < self.config["max_retries"] - 1:
                    time.sleep(self.config["retry_delay"])
                else:
                    logger.debug(f"下载失败 [{index}]: {url} - {str(e)[:100]}")
                    return None
            except Exception as e:
                logger.error(f"未知错误 [{index}]: {url} - {e}")
                return None
        
        return None
    
    def _read_csv_in_batches(self):
        """分批读取CSV文件，避免内存溢出"""
        csv_file = Path(self.config["csv_file"])
        batch = []
        index = 0
        
        with open(csv_file, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                url = row.get('image_url', '').strip()
                caption = row.get('caption', '').strip()
                
                if url:
                    batch.append((index, url, caption))
                    index += 1
                
                if len(batch) >= self.config["batch_size"]:
                    yield batch
                    batch = []
            
            # 处理剩余的数据
            if batch:
                yield batch
    
    def download_all(self):
        """下载所有图片"""
        self.start_time = time.time()
        total_processed = 0
        total_success = 0
        total_skipped = 0
        
        logger.info("=" * 60)
        logger.info("开始下载图片数据集")
        logger.info(f"输出目录: {self.output_dir.absolute()}")
        logger.info(f"并发线程: {self.config['max_workers']}")
        logger.info(f"已完成数量: {len(self.completed)}")
        logger.info("=" * 60)
        
        try:
            for batch_num, batch in enumerate(self._read_csv_in_batches(), 1):
                logger.info(f"处理批次 {batch_num}，包含 {len(batch)} 条记录...")
                
                # 过滤掉已下载的
                to_download = [(idx, url, cap) for idx, url, cap in batch if url not in self.completed]
                skipped = len(batch) - len(to_download)
                total_skipped += skipped
                
                if not to_download:
                    logger.info(f"批次 {batch_num} 全部已下载，跳过")
                    continue
                
                logger.info(f"需要下载: {len(to_download)} 个, 跳过: {skipped} 个")
                
                # 多线程下载
                with ThreadPoolExecutor(max_workers=self.config["max_workers"]) as executor:
                    futures = {
                        executor.submit(self._download_single, url, idx, cap): (idx, url)
                        for idx, url, cap in to_download
                    }
                    
                    for future in as_completed(futures):
                        idx, url = futures[future]
                        total_processed += 1
                        
                        try:
                            result = future.result()
                            if result:
                                self.completed.add(result)
                                total_success += 1
                                self.download_count += 1
                                
                                # 定期保存进度
                                if self.download_count % self.config["save_interval"] == 0:
                                    self._save_progress()
                                    self._print_progress(total_processed, total_success)
                            else:
                                if url not in self.completed:
                                    self.failed_urls.append(url)
                        except Exception as e:
                            logger.error(f"处理结果时出错: {e}")
                            self.failed_urls.append(url)
                
                # 每批次后保存进度
                self._save_progress()
                
        except KeyboardInterrupt:
            logger.warning("\n用户中断下载，正在保存进度...")
            self._save_progress()
            self._save_failed()
            raise
        
        # 最终保存
        self._save_progress()
        self._save_failed()
        self._print_final_stats(total_processed, total_success, total_skipped)
    
    def _print_progress(self, processed: int, success: int):
        """打印进度信息"""
        elapsed = time.time() - self.start_time
        speed = success / elapsed if elapsed > 0 else 0
        logger.info(
            f"进度: 已处理 {processed} | 成功 {success} | "
            f"总完成 {len(self.completed)} | 速度 {speed:.1f} 张/秒"
        )
    
    def _print_final_stats(self, processed: int, success: int, skipped: int):
        """打印最终统计信息"""
        elapsed = time.time() - self.start_time
        
        logger.info("=" * 60)
        logger.info("下载完成！")
        logger.info(f"总处理数: {processed}")
        logger.info(f"成功下载: {success}")
        logger.info(f"跳过(已存在): {skipped}")
        logger.info(f"失败数: {len(self.failed_urls)}")
        logger.info(f"总耗时: {elapsed/3600:.2f} 小时")
        logger.info(f"平均速度: {success/elapsed:.1f} 张/秒" if elapsed > 0 else "")
        logger.info(f"图片保存位置: {self.output_dir.absolute()}")
        logger.info("=" * 60)
    
    def _save_failed(self):
        """保存失败记录"""
        if self.failed_urls:
            failed_file = Path(self.config["failed_file"])
            with open(failed_file, 'w', encoding='utf-8') as f:
                for url in self.failed_urls:
                    f.write(url + '\n')
            logger.info(f"失败记录已保存到: {failed_file}")


def main():
    """主函数"""
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║            大规模图片数据集下载工具                          ║
    ║  支持断点续传 | 多线程下载 | 自动重试                        ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # 检查CSV文件
    if not Path(CONFIG["csv_file"]).exists():
        logger.error(f"CSV文件不存在: {CONFIG['csv_file']}")
        return
    
    # 创建下载器并开始下载
    downloader = ImageDownloader(CONFIG)
    
    try:
        downloader.download_all()
    except KeyboardInterrupt:
        print("\n\n下载已中断。进度已保存，下次运行将继续下载。")


if __name__ == "__main__":
    main()
