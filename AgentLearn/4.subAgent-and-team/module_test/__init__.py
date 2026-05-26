import sys
from pathlib import Path

STAGE_ROOT = Path(__file__).resolve().parents[1]
# 测试包内模块需要导入阶段根目录下的 mcp_client、tools 等模块。
stage_root_str = str(STAGE_ROOT)
if stage_root_str in sys.path:
    sys.path.remove(stage_root_str)
sys.path.insert(0, stage_root_str)
