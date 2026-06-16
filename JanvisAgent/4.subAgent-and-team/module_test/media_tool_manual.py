# encoding: utf-8
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from tools.tool_manager import AgentToolHandlers, ToolManager, ToolManagerConfig
from tools.tool_names import ToolNameConstant


class DummyClient:
	pass


def test_audio_transcribe_gaia_sample():
	audio_path = (
		"D:\\AgentLongTaskTest\\GAIA\\2023\\"
		"\u5206\u7c7bvalidation\\Level_1\\1f975693-876d-457b-a649-393859e79bf3.mp3"
	)
	manager = ToolManager(
		config=ToolManagerConfig(
			project_root=str(PROJECT_ROOT),
			client=DummyClient(),
			model="manual-test",
			temperature=0,
			is_main_agent=False,
		),
		handlers=AgentToolHandlers(),
		mcp_client=None,
	)

	result = json.loads(
		manager.available_functions[ToolNameConstant.AUDIO_TRANSCRIBE](
			path=audio_path,
			output_format="txt",
		)
	)
	print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
	assert result.get("ok") is True
	assert result.get("content")
