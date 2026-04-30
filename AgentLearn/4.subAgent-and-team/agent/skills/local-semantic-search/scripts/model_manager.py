#!/usr/bin/env python3
"""
模型管理脚本（纯本地模式）
扫描本地模型、切换模型、配置新模型
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from typing import List, Dict, Optional

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


SKILL_DIR = Path(__file__).parent.parent
CONFIG_FILE = SKILL_DIR / "model_config.json"
MODEL_DIR = SKILL_DIR / "models"
HUGGINGFACE_CACHE = Path.home() / '.cache' / 'huggingface' / 'hub'

DEFAULT_MODEL = 'moka-ai/m3e-base'


def scan_local_models() -> List[Dict]:
    """扫描本地缓存的模型（HuggingFace缓存 + skill本地模型目录）"""
    models = []
    
    # 1. 扫描HuggingFace缓存
    if HUGGINGFACE_CACHE.exists():
        for item in HUGGINGFACE_CACHE.iterdir():
            if item.is_dir() and item.name.startswith('models--'):
                model_name = item.name.replace('models--', '').replace('--', '/')
                
                snapshots_dir = item / 'snapshots'
                if snapshots_dir.exists():
                    versions = []
                    for snapshot in snapshots_dir.iterdir():
                        if snapshot.is_dir():
                            versions.append({
                                'version': snapshot.name,
                                'path': str(snapshot),
                                'source': 'huggingface_cache'
                            })
                
                if versions:
                    models.append({
                        'model_name': model_name,
                        'cache_name': item.name,
                        'versions': versions,
                        'source': 'huggingface_cache',
                        'is_local': True
                    })
    
    # 2. 扫描skill本地模型目录
    if MODEL_DIR.exists():
        for item in MODEL_DIR.iterdir():
            if item.is_dir():
                models.append({
                    'model_name': item.name,
                    'cache_name': item.name,
                    'versions': [{
                        'version': 'local',
                        'path': str(item),
                        'source': 'skill_local'
                    }],
                    'source': 'skill_local',
                    'is_local': True
                })
    
    return models


def get_current_model() -> Dict:
    """获取当前配置的模型"""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    
    # 默认查找m3e-base缓存
    default_path = find_model_in_cache(DEFAULT_MODEL)
    return {
        'model_name': DEFAULT_MODEL,
        'local_path': default_path,
        'source': 'huggingface_cache' if default_path else 'unknown'
    }


def find_model_in_cache(model_name: str) -> Optional[str]:
    """在缓存中查找模型"""
    # 查找HuggingFace缓存
    cache_name = f"models--{model_name.replace('/', '--')}"
    cache_dir = HUGGINGFACE_CACHE / cache_name / 'snapshots'
    
    if cache_dir.exists():
        for snapshot in cache_dir.iterdir():
            if snapshot.is_dir():
                return str(snapshot)
    
    # 查找skill本地目录
    local_dir = MODEL_DIR / model_name.replace('/', '--')
    if local_dir.exists():
        return str(local_dir)
    
    return None


def save_model_config(model_name: str, local_path: str, source: str = 'huggingface_cache'):
    """保存模型配置"""
    config = {
        'model_name': model_name,
        'local_path': local_path,
        'source': source,
        'updated_at': str(Path.cwd())
    }
    
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 模型配置已保存: {CONFIG_FILE}")
    print(f"✓ 当前模型: {model_name}")
    print(f"✓ 本地路径: {local_path}")


def test_model_load(model_path: str) -> bool:
    """测试模型是否可以加载"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("✗ sentence-transformers 未安装", file=sys.stderr)
        return False
    
    try:
        print(f"正在测试加载模型: {model_path}")
        model = SentenceTransformer(model_path)
        print(f"✓ 模型加载成功！")
        print(f"  维度: {model.get_sentence_embedding_dimension()}")
        print(f"  最大序列长度: {model.max_seq_length}")
        return True
    except Exception as e:
        print(f"✗ 模型加载失败: {e}", file=sys.stderr)
        return False


def format_local_models(models: List[Dict]) -> str:
    """格式化本地模型列表"""
    if not models:
        return "✗ 未找到本地模型\n\n请将模型权重放置到以下目录之一:\n" + \
               f"  1. HuggingFace缓存: {HUGGINGFACE_CACHE}\n" + \
               f"  2. Skill本地目录: {MODEL_DIR}"
    
    output = []
    output.append(f"\n本地模型列表 (共 {len(models)} 个):")
    output.append("=" * 60)
    
    current_config = get_current_model()
    current_path = current_config.get('local_path', '')
    
    for i, model in enumerate(models, 1):
        current_mark = "【当前】" if any(v['path'] == current_path for v in model['versions']) else ""
        source_mark = "📁" if model['source'] == 'skill_local' else "💾"
        
        output.append(f"\n[{i}] {source_mark} {current_mark}{model['model_name']}")
        output.append(f"    来源: {model['source']}")
        output.append(f"    版本数: {len(model['versions'])}")
        for v in model['versions']:
            mark = "✓ 当前使用" if v['path'] == current_path else ""
            output.append(f"      - {v['version']} {mark}")
    
    return '\n'.join(output)


def interactive_model_selection(models: List[Dict]) -> Optional[Dict]:
    """交互式模型选择"""
    if not models:
        print("\n✗ 未找到本地模型")
        print("\n请将模型权重放置到以下目录之一:")
        print(f"  1. HuggingFace缓存: {HUGGINGFACE_CACHE}")
        print(f"  2. Skill本地目录: {MODEL_DIR}")
        print("\n详见: python scripts/model_manager.py --guide")
        return None
    
    print(format_local_models(models))
    
    print("\n请选择要使用的模型 (输入序号，或输入 'q' 退出):")
    
    try:
        choice = input("> ").strip()
        
        if choice.lower() == 'q':
            return None
        
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            selected = models[idx]
            
            if len(selected['versions']) > 1:
                print(f"\n该模型有 {len(selected['versions'])} 个版本:")
                for i, v in enumerate(selected['versions'], 1):
                    print(f"  [{i}] {v['version']} ({v['source']})")
                print("\n请选择版本 (输入序号):")
                ver_choice = input("> ").strip()
                ver_idx = int(ver_choice) - 1
                if 0 <= ver_idx < len(selected['versions']):
                    return {
                        'model_name': selected['model_name'],
                        'path': selected['versions'][ver_idx]['path'],
                        'source': selected['versions'][ver_idx]['source']
                    }
            
            return {
                'model_name': selected['model_name'],
                'path': selected['versions'][0]['path'],
                'source': selected['versions'][0]['source']
            }
        else:
            print("✗ 无效的选择")
            return None
    except Exception as e:
        print(f"✗ 输入错误: {e}")
        return None


def show_download_guide():
    """显示下载新模型的指导"""
    guide = """
================================================================================
                        新模型下载与配置指南
================================================================================

一、下载模型权重

方法1: 使用 HuggingFace CLI (推荐)
------------------------------------------
pip install huggingface_hub
huggingface-cli download moka-ai/m3e-base --local-dir ~/.cache/huggingface/hub/models--moka-ai--m3e-base/snapshots/v1

方法2: 使用 Python (需要网络)
------------------------------------------
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('moka-ai/m3e-base')
# 模型会自动缓存到 ~/.cache/huggingface/hub/

方法3: 手动下载
------------------------------------------
1. 访问 HuggingFace: https://huggingface.co/models?search=sentence-transformers
2. 找到目标模型页面，如: https://huggingface.co/moka-ai/m3e-base
3. 点击 "Files and versions" 下载所有文件
4. 放置到缓存目录或skill本地目录

二、模型存放位置

位置1: HuggingFace缓存目录 (自动检测)
------------------------------------------
路径: ~/.cache/huggingface/hub/models--<模型名>/snapshots/<版本号>/
示例: ~/.cache/huggingface/hub/models--moka-ai--m3e-base/snapshots/xxx/

位置2: Skill本地模型目录 (推荐自定义模型)
------------------------------------------
路径: {model_dir}
示例: {model_dir}/my-custom-model/

三、切换到新模型

步骤1: 扫描本地模型
------------------------------------------
python scripts/model_manager.py --scan

步骤2: 交互式选择模型
------------------------------------------
python scripts/model_manager.py --interactive

步骤3: 测试模型加载
------------------------------------------
python scripts/model_manager.py --test <模型路径>

四、常用模型推荐

| 模型名称 | 说明 | 维度 |
|----------|------|------|
| moka-ai/m3e-base | 中文优化，推荐 | 768 |
| BAAI/bge-large-zh | 中文高精度 | 1024 |
| BAAI/bge-small-zh | 中文快速 | 512 |
| sentence-transformers/all-MiniLM-L6-v2 | 英文 | 384 |

================================================================================
"""
    print(guide.format(model_dir=str(MODEL_DIR)))


def main():
    parser = argparse.ArgumentParser(description='模型管理工具（纯本地模式）')
    parser.add_argument('--scan', '-s', action='store_true', help='扫描本地模型')
    parser.add_argument('--current', '-c', action='store_true', help='显示当前模型')
    parser.add_argument('--switch', '-w', type=str, help='切换到指定模型路径')
    parser.add_argument('--test', '-t', type=str, help='测试模型加载')
    parser.add_argument('--interactive', '-i', action='store_true', help='交互式选择模型')
    parser.add_argument('--guide', '-g', action='store_true', help='显示下载配置指南')
    parser.add_argument('--json', '-j', action='store_true', help='输出JSON格式')
    
    args = parser.parse_args()
    
    if args.guide:
        show_download_guide()
        return
    
    if args.scan:
        models = scan_local_models()
        if args.json:
            print(json.dumps(models, ensure_ascii=False, indent=2))
        else:
            print(format_local_models(models))
        return
    
    if args.current:
        config = get_current_model()
        if args.json:
            print(json.dumps(config, ensure_ascii=False, indent=2))
        else:
            print(f"当前模型: {config['model_name']}")
            print(f"本地路径: {config['local_path']}")
            print(f"来源: {config['source']}")
        return
    
    if args.test:
        success = test_model_load(args.test)
        if success:
            print("\n可以使用以下命令切换到此模型:")
            print(f"  python scripts/model_manager.py --switch {args.test}")
        return
    
    if args.switch:
        if test_model_load(args.switch):
            model_name = Path(args.switch).name
            save_model_config(model_name, args.switch, 'manual')
        return
    
    if args.interactive:
        models = scan_local_models()
        selected = interactive_model_selection(models)
        if selected:
            if test_model_load(selected['path']):
                save_model_config(selected['model_name'], selected['path'], selected['source'])
        return
    
    # 默认显示帮助
    print("模型管理工具（纯本地模式）")
    print("\n常用命令:")
    print("  --scan          扫描本地模型")
    print("  --current       显示当前使用的模型")
    print("  --interactive   交互式选择模型")
    print("  --test          测试模型加载")
    print("  --switch        切换到指定模型路径")
    print("  --guide         显示下载配置指南")
    print("\n示例:")
    print("  python scripts/model_manager.py --scan")
    print("  python scripts/model_manager.py --interactive")
    print("  python scripts/model_manager.py --guide")


if __name__ == '__main__':
    main()
