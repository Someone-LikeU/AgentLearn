import sys
from pathlib import Path

STAGE_ROOT = Path(__file__).resolve().parents[1]
# 测试包内模块需要导入阶段根目录下的 mcp_client、tools 等模块。
if str(STAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE_ROOT))
