#!/usr/bin/env python3
"""
向量库重置脚本
清除本地向量库的所有数据，恢复为初始状态
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path


SKILL_DIR = Path(__file__).parent.parent
VECTOR_STORE_DIR = SKILL_DIR / "vector_store"
INDEX_METADATA_FILE = VECTOR_STORE_DIR / "metadata.json"


def reset_vector_store(force: bool = False) -> dict:
    """重置向量库"""
    result = {
        'success': False,
        'message': '',
        'deleted_files': [],
        'deleted_dirs': []
    }
    
    if not VECTOR_STORE_DIR.exists():
        result['success'] = True
        result['message'] = '向量库目录不存在，无需重置'
        return result
    
    if not force:
        result['message'] = '需要使用 --force 参数确认重置操作'
        return result
    
    try:
        for item in VECTOR_STORE_DIR.iterdir():
            if item.is_file():
                result['deleted_files'].append(str(item.name))
                item.unlink()
            elif item.is_dir():
                result['deleted_dirs'].append(str(item.name))
                shutil.rmtree(item)
        
        VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        
        result['success'] = True
        result['message'] = '向量库已重置，恢复为初始状态'
        
    except Exception as e:
        result['message'] = f'重置失败: {str(e)}'
    
    return result


def check_vector_store_status() -> dict:
    """检查向量库状态"""
    status = {
        'exists': VECTOR_STORE_DIR.exists(),
        'path': str(VECTOR_STORE_DIR),
        'files': [],
        'dirs': [],
        'total_size': 0,
        'has_metadata': INDEX_METADATA_FILE.exists(),
        'indexed_files': 0,
        'indexed_dir': None
    }
    
    if status['exists']:
        for item in VECTOR_STORE_DIR.iterdir():
            if item.is_file():
                status['files'].append({
                    'name': item.name,
                    'size': item.stat().st_size
                })
                status['total_size'] += item.stat().st_size
            elif item.is_dir():
                status['dirs'].append(item.name)
        
        if INDEX_METADATA_FILE.exists():
            try:
                with open(INDEX_METADATA_FILE, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                status['indexed_files'] = metadata.get('total_files', 0)
                status['indexed_dir'] = metadata.get('root_dir', None)
                status['indexed_at'] = metadata.get('indexed_at', None)
            except Exception:
                pass
    
    return status


def format_status_output(status: dict) -> str:
    """格式化状态输出"""
    if not status['exists']:
        return "向量库目录不存在（初始状态）"
    
    output = []
    output.append(f"向量库路径: {status['path']}")
    output.append(f"总大小: {status['total_size'] / 1024 / 1024:.2f} MB")
    output.append(f"文件数: {len(status['files'])}")
    output.append(f"目录数: {len(status['dirs'])}")
    
    if status['has_metadata']:
        output.append(f"已索引文件数: {status['indexed_files']}")
        output.append(f"索引目录: {status['indexed_dir'] or 'N/A'}")
        output.append(f"索引时间: {status['indexed_at'] or 'N/A'}")
    else:
        output.append("无索引元数据")
    
    if status['files']:
        output.append("\n文件列表:")
        for f in status['files']:
            output.append(f"  - {f['name']} ({f['size'] / 1024:.1f} KB)")
    
    return '\n'.join(output)


def format_reset_output(result: dict) -> str:
    """格式化重置输出"""
    if not result['success']:
        return f"重置失败: {result['message']}"
    
    output = []
    output.append(result['message'])
    
    if result['deleted_files']:
        output.append(f"删除文件: {len(result['deleted_files'])} 个")
        for f in result['deleted_files']:
            output.append(f"  - {f}")
    
    if result['deleted_dirs']:
        output.append(f"删除目录: {len(result['deleted_dirs'])} 个")
        for d in result['deleted_dirs']:
            output.append(f"  - {d}")
    
    output.append("\n向量库已恢复为初始状态，可以重新构建索引")
    
    return '\n'.join(output)


def main():
    parser = argparse.ArgumentParser(description='向量库重置工具')
    parser.add_argument('--status', '-s', action='store_true', help='查看向量库状态')
    parser.add_argument('--reset', '-r', action='store_true', help='重置向量库')
    parser.add_argument('--force', '-f', action='store_true', help='确认重置操作（必须）')
    parser.add_argument('--json', '-j', action='store_true', help='输出JSON格式')
    
    args = parser.parse_args()
    
    if args.status:
        status = check_vector_store_status()
        if args.json:
            print(json.dumps(status, ensure_ascii=False, indent=2))
        else:
            print(format_status_output(status))
    
    elif args.reset:
        if not args.force:
            print("警告: 重置操作将删除所有向量库数据！")
            print("请使用 --force 参数确认操作:")
            print("  python scripts/reset_vector_store.py --reset --force")
            if args.json:
                result = {'success': False, 'message': '需要 --force 参数确认'}
                print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        
        result = reset_vector_store(force=True)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(format_reset_output(result))
    
    else:
        # 默认显示状态
        status = check_vector_store_status()
        if args.json:
            print(json.dumps(status, ensure_ascii=False, indent=2))
        else:
            print(format_status_output(status))
            print("\n使用方法:")
            print("  查看状态: python scripts/reset_vector_store.py --status")
            print("  重置向量库: python scripts/reset_vector_store.py --reset --force")


if __name__ == '__main__':
    main()
