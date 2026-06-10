"""Tool package for the sub-agent team stage."""

import sys
from pathlib import Path

STAGE_ROOT = Path(__file__).resolve().parents[1]
# 子包内模块需要导入阶段根目录下的 prompt_loader、mcp_client 等模块。
if str(STAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE_ROOT))
