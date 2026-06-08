# encoding: utf-8
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_Teams import Agent
from multi_agent import build_table_game_runtime_config


class AgentInitializationProfilesTest(unittest.TestCase):
    def _build_agent(self, **kwargs):
        defaults = {
            "model": "test-model",
            "base_url": "http://127.0.0.1:9/v1",
            "api_key": "test-key",
        }
        defaults.update(kwargs)
        # 初始化测试只验证 prompt 和工具暴露情况，不触发模型请求。
        with redirect_stdout(io.StringIO()):
            return Agent(**defaults)

    def init_agent_test(self):
        main_agent = self._build_agent(name="主Agent")
        sub_agent = self._build_agent(
            role="测试子Agent",
            name="子Agent",
            is_main_agent=False,
        )
        table_game_agent = self._build_agent(
            role="蓝方忠臣",
            name="玩家A",
            is_main_agent=False,
            runtime_config=build_table_game_runtime_config(),
        )

        main_prompt = main_agent._cached_system_prompt
        sub_prompt = sub_agent._cached_system_prompt
        table_game_prompt = table_game_agent._cached_system_prompt

        self.assertIn("daily tasks or software engineering tasks", main_prompt)
        self.assertIn("Use the instructions below", main_prompt)
        self.assertGreater(len(main_agent._all_tools), 0)
        self.assertIsNotNone(main_agent.memory_manager)

        self.assertIn("You are a 测试子Agent", sub_prompt)
        self.assertIn("Use the instructions below", sub_prompt)
        self.assertGreater(len(sub_agent._all_tools), 0)
        self.assertIsNone(sub_agent.memory_manager)

        self.assertIn("桌游模拟 Agent", table_game_prompt)
        self.assertIn("阿瓦隆游戏规则", table_game_prompt)
        self.assertIn("蓝方忠臣", table_game_prompt)
        self.assertIn("玩家A", table_game_prompt)
        self.assertEqual(table_game_agent._all_tools, [])
        self.assertIsNone(table_game_agent.memory_manager)
        self.assertNotIn("software engineering tasks", table_game_prompt)

        return {
            "main_prompt": main_prompt,
            "sub_prompt": sub_prompt,
            "table_game_prompt": table_game_prompt,
        }

    def test_init_agent_test(self):
        self.init_agent_test()


if __name__ == "__main__":
    unittest.main()
