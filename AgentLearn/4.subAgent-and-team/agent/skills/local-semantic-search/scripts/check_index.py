#!/usr/bin/env python3
"""
索引状态检查脚本
检查指定目录是否已向量化，并返回索引状态信息
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


SKILL_DIR = Path(__file__).parent.parent
VECTOR_STORE_DIR = SKILL_DIR / "vector_store"
INDEX_METADATA_FILE = VECTOR_STORE_DIR / "metadata.json"
COLLECTION_NAME = "local_files"

DEFAULT_FILE_TYPES = ['txt', 'md', 'pdf', 'docx', 'pptx']


def count_files_in_dir(dir_path: Path, file_types: list) -> int:
    """统计目录中指定类型的文件数量"""
    count = 0
    for ext in file_types:
        pattern = f'*.{ext}'
        for _ in dir_path.rglob(pattern):
            if _.is_file():
                count += 1
    return count


def check_index_status(target_dir: Path) -> dict:
    """检查目录索引状态"""
    result = {
        'indexed': False,
        'dir': str(target_dir.absolute()),
        'total_files': 0,
        'indexed_files': 0,
        'indexed_at': None,
        'file_types': DEFAULT_FILE_TYPES,
        'status': 'not_indexed'
    }
    
    if not target_dir.exists():
        result['error'] = '目录不存在'
        return result
    
    result['total_files'] = count_files_in_dir(target_dir, DEFAULT_FILE_TYPES)
    
    if not INDEX_METADATA_FILE.exists():
        result['status'] = 'not_indexed'
        return result
    
    try:
        with open(INDEX_METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        indexed_dir = Path(metadata.get('root_dir', ''))
        
        if indexed_dir.resolve() == target_dir.resolve():
            result['indexed'] = True
            result['indexed_files'] = metadata.get('total_files', 0)
            result['indexed_at'] = metadata.get('indexed_at', None)
            result['file_types'] = metadata.get('file_types', DEFAULT_FILE_TYPES)
            
            if result['indexed_files'] >= result['total_files']:
                result['status'] = 'complete'
            elif result['indexed_files'] > 0:
                result['status'] = 'partial'
            else:
                result['status'] = 'empty'
        else:
            result['indexed_dir'] = str(indexed_dir)
            result['status'] = 'different_dir'
            result['message'] = f'当前索引的是其他目录: {indexed_dir}'
        
        if CHROMADB_AVAILABLE:
            try:
                client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
                collection = client.get_collection(name=COLLECTION_NAME)
                result['vector_count'] = collection.count()
            except Exception:
                result['vector_count'] = 0
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def format_output(result: dict) -> str:
    """格式化输出结果"""
    if 'error' in result:
        return f"错误: {result['error']}"
    
    if not result['indexed']:
        if result['status'] == 'different_dir':
            return f"目录未索引。当前索引的是: {result.get('indexed_dir', 'N/A')}\n目录文件数: {result['total_files']}"
        return f"目录未索引\n目录文件数: {result['total_files']}\n需要构建索引才能搜索"
    
    status_text = {
        'complete': '索引完整',
        'partial': '索引部分完成',
        'empty': '索引为空',
        'stale': '索引可能过期'
    }
    
    output = []
    output.append(f"目录: {result['dir']}")
    output.append(f"状态: {status_text.get(result['status'], result['status'])}")
    output.append(f"目录文件数: {result['total_files']}")
    output.append(f"已索引文件数: {result['indexed_files']}")
    output.append(f"索引时间: {result['indexed_at'] or 'N/A'}")
    output.append(f"向量数: {result.get('vector_count', 'N/A')}")
    output.append(f"文件类型: {', '.join(result['file_types'])}")
    
    return '\n'.join(output)


def main():
    parser = argparse.ArgumentParser(description='检查目录索引状态')
    parser.add_argument('--dir', '-d', type=str, required=True, help='要检查的目录路径')
    parser.add_argument('--json', '-j', action='store_true', help='输出JSON格式')
    
    args = parser.parse_args()
    
    target_dir = Path(args.dir).resolve()
    result = check_index_status(target_dir)
    
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_output(result))


if __name__ == '__main__':
    main()
