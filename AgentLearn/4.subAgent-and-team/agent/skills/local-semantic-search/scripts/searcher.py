#!/usr/bin/env python3
"""
本地文件语义搜索器
基于向量索引执行语义搜索
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


SKILL_DIR = Path(__file__).parent.parent
VECTOR_STORE_DIR = SKILL_DIR / "vector_store"
INDEX_METADATA_FILE = VECTOR_STORE_DIR / "metadata.json"
MODEL_CONFIG_FILE = SKILL_DIR / "model_config.json"
COLLECTION_NAME = "local_files"

DEFAULT_MODEL = 'moka-ai/m3e-base'
HUGGINGFACE_CACHE = Path.home() / '.cache' / 'huggingface' / 'hub'


def load_model_config() -> Dict:
    """加载模型配置"""
    if MODEL_CONFIG_FILE.exists():
        try:
            with open(MODEL_CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {'model_name': DEFAULT_MODEL, 'local_path': None, 'use_local': True}


def find_local_model_path(model_name: str) -> Optional[str]:
    """查找本地模型路径"""
    cache_name = f"models--{model_name.replace('/', '--')}"
    cache_dir = HUGGINGFACE_CACHE / cache_name / 'snapshots'
    
    if cache_dir.exists():
        for snapshot in cache_dir.iterdir():
            if snapshot.is_dir():
                return str(snapshot)
    return None


def load_model_with_fallback():
    """加载模型（纯本地模式），失败时提供解决方案"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return None, "sentence-transformers 未安装，请运行: pip install sentence-transformers"
    
    config = load_model_config()
    model_name = config.get('model_name', DEFAULT_MODEL)
    local_path = config.get('local_path')
    
    print(f"正在加载嵌入模型: {model_name}")
    
    # 纯本地模式：只从本地路径加载
    try:
        # 1. 优先使用配置中的本地路径
        if local_path and Path(local_path).exists():
            model = SentenceTransformer(local_path)
            return model, f"✓ 成功加载本地模型: {local_path}"
        
        # 2. 从HuggingFace缓存查找
        cache_path = find_local_model_path(model_name)
        if cache_path:
            model = SentenceTransformer(cache_path)
            return model, f"✓ 成功加载缓存模型: {cache_path}"
        
        # 3. 未找到模型
        error_msg = f"✗ 未找到本地模型: {model_name}\n"
        error_msg += "\n解决方案:\n"
        error_msg += "1. 扫描本地模型:\n"
        error_msg += "   python scripts/model_manager.py --scan\n"
        error_msg += "2. 交互式选择模型:\n"
        error_msg += "   python scripts/model_manager.py --interactive\n"
        error_msg += "3. 查看下载配置指南:\n"
        error_msg += "   python scripts/model_manager.py --guide\n"
        return None, error_msg
        
    except Exception as e:
        error_msg = f"✗ 模型加载失败: {e}\n"
        error_msg += "\n解决方案:\n"
        error_msg += "1. 扫描本地模型:\n"
        error_msg += "   python scripts/model_manager.py --scan\n"
        error_msg += "2. 交互式选择模型:\n"
        error_msg += "   python scripts/model_manager.py --interactive\n"
        error_msg += "3. 查看下载配置指南:\n"
        error_msg += "   python scripts/model_manager.py --guide\n"
        return None, error_msg


def load_metadata() -> Optional[Dict]:
    """加载索引元数据"""
    if not INDEX_METADATA_FILE.exists():
        return None
    
    try:
        with open(INDEX_METADATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载元数据失败: {e}", file=sys.stderr)
        return None


def search(query: str, top_k: int = 10, threshold: float = 0.0) -> Dict:
    """执行语义搜索"""
    start_time = time.time()
    
    if not CHROMADB_AVAILABLE:
        return {
            'error': 'chromadb 未安装',
            'results': [],
            'total': 0,
            'search_time_ms': 0
        }
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return {
            'error': 'sentence-transformers 未安装',
            'results': [],
            'total': 0,
            'search_time_ms': 0
        }
    
    metadata = load_metadata()
    if not metadata:
        return {
            'error': '索引不存在，请先运行 indexer.py 构建索引',
            'results': [],
            'total': 0,
            'search_time_ms': 0
        }
    
    try:
        model, message = load_model_with_fallback()
        if model is None:
            return {
                'error': message,
                'results': [],
                'total': 0,
                'search_time_ms': int((time.time() - start_time) * 1000)
            }
        
        print(message)
        
        print("正在连接向量数据库...")
        client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
        collection = client.get_collection(name=COLLECTION_NAME)
        
        print(f"正在搜索: {query}")
        query_embedding = model.encode(query, show_progress_bar=False).tolist()
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        search_results = []
        if results and results['metadatas']:
            for i, meta in enumerate(results['metadatas'][0]):
                distance = results['distances'][0][i] if results.get('distances') else 0
                score = 1 - distance
                
                if score >= threshold:
                    search_results.append({
                        'path': meta.get('path', ''),
                        'name': meta.get('name', ''),
                        'type': meta.get('type', ''),
                        'size': int(meta.get('size', 0)),
                        'mtime': meta.get('mtime', ''),
                        'score': round(score, 4),
                        'matches': [results['documents'][0][i][:200]] if results.get('documents') else []
                    })
        
        search_time_ms = int((time.time() - start_time) * 1000)
        
        return {
            'results': search_results,
            'total': len(search_results),
            'search_time_ms': search_time_ms,
            'indexed_dir': metadata.get('root_dir', ''),
            'indexed_at': metadata.get('indexed_at', ''),
            'total_indexed_files': metadata.get('total_files', 0)
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'results': [],
            'total': 0,
            'search_time_ms': int((time.time() - start_time) * 1000)
        }


def format_output(result: Dict) -> str:
    """格式化输出结果"""
    if 'error' in result:
        return f"错误: {result['error']}"
    
    output = []
    output.append(f"搜索完成，耗时 {result['search_time_ms']}ms")
    output.append(f"索引目录: {result.get('indexed_dir', 'N/A')}")
    output.append(f"索引文件数: {result.get('total_indexed_files', 0)}")
    output.append(f"找到 {result['total']} 个结果:\n")
    
    for i, item in enumerate(result['results'], 1):
        output.append(f"[{i}] {item['name']}")
        output.append(f"    路径: {item['path']}")
        output.append(f"    类型: {item['type']} | 大小: {item['size']} bytes | 相似度: {item['score']:.2%}")
        output.append(f"    修改时间: {item['mtime']}")
        if item.get('matches'):
            output.append(f"    匹配内容: {item['matches'][0][:100]}...")
        output.append("")
    
    return '\n'.join(output)


def main():
    parser = argparse.ArgumentParser(description='本地文件语义搜索器')
    parser.add_argument('--query', '-q', type=str, required=True, help='搜索查询文本')
    parser.add_argument('--top-k', '-k', type=int, default=10, help='返回结果数量')
    parser.add_argument('--threshold', '-t', type=float, default=0.0, help='相似度阈值 (0-1)')
    parser.add_argument('--json', action='store_true', help='输出 JSON 格式')
    
    args = parser.parse_args()
    
    result = search(args.query, args.top_k, args.threshold)
    
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_output(result))


if __name__ == '__main__':
    main()
