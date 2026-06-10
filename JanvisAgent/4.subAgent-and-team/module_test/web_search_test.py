# encoding : utf-8
# @Time    : 2026/5/25 16:54

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 直接运行 module_test 下的脚本时，Python 不会自动把上一层阶段目录加入导入路径。
project_root_str = str(PROJECT_ROOT)
if project_root_str in sys.path:
	sys.path.remove(project_root_str)
sys.path.insert(0, project_root_str)

from module_test.test_tool_manager import debug_web_search

if __name__ == '__main__':
	debug_web_search("F1方程式赛车2023赛季新加坡站", 5)
