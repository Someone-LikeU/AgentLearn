# encoding: utf-8
import json
import os
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

from prompt_loader import load_prompt


AgentFactory = Callable[[str, str], Any]


def _read_model_config() -> dict:
    config_path = Path(__file__).resolve().parent / "agent" / "config" / "model_config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {"models": []}
    if not isinstance(config, dict):
        return {"models": []}
    if not isinstance(config.get("models"), list):
        config["models"] = []
    return config


def _resolve_model_connection(model_name: str) -> tuple[str | None, str | None, bool, list[str]]:
    # 团队规划器也按模型名读取连接配置，避免 plan_team 使用默认 OPENAI_API_KEY。
    for model_info in _read_model_config().get("models", []):
        if not isinstance(model_info, dict) or model_info.get("name") != model_name:
            continue

        missing_env_vars: list[str] = []
        base_url_env = model_info.get("base_url_env")
        if base_url_env:
            base_url = os.environ.get(str(base_url_env))
            if base_url is None:
                missing_env_vars.append(str(base_url_env))
        else:
            base_url = model_info.get("base_url")

        api_key_env = model_info.get("api_key_env")
        if api_key_env:
            api_key = os.environ.get(str(api_key_env))
            if api_key is None:
                missing_env_vars.append(str(api_key_env))
        else:
            api_key = model_info.get("api_key")

        return base_url, api_key, True, missing_env_vars

    return None, None, False, []


class Team:
    def __init__(self, agent_factory: AgentFactory | None = None):
        self.agents: dict[str, Any] = {}
        self.agent_factory = agent_factory or self._default_agent_factory

    def _default_agent_factory(self, name: str, role: str):
        # 延迟导入 Agent，避免 agent_Teams 兼容导出 Team 时形成循环导入。
        from agent_Teams import Agent

        return Agent(role=role, name=name, is_main_agent=False)

    def hire(self, name: str, role: str):
        agent = self.agent_factory(name, role)
        self.agents[name] = agent
        return agent

    def send(self, from_name: str, to_name: str, message: str):
        if to_name not in self.agents:
            return f"Error: {to_name} not found"
        self.agents[to_name].receive(from_name, message)
        print(f"  [communication] from {from_name} to {to_name}, message is: {message[:60]}...")
        return "OK"

    def broadcast(self, from_name: str, message: str):
        for name, agent in self.agents.items():
            if name != from_name:
                agent.receive(from_name, message)
        print(f"  [broadcast] from {from_name} to all teammates: {message[:60]}...")
        return "OK"

    def disband(self):
        names = list(self.agents.keys())
        self.agents.clear()
        print(f"  [dismiss] The team are dismissed ({', '.join(names)})")


class TeamOrchestrator:
    def __init__(
            self,
            model: str = "qwen3.5:9b",
            temperature: float = 0.1,
            base_url: str | None = None,
            api_key: str | None = None,
            client=None,
            mcp_client=None,
            agent_runtime_config=None,
            agent_is_main_agent: bool = False,
            agent_class=None,
    ):
        self.model = model
        self.temperature = temperature
        # 成员 Agent 只在显式传入连接信息时使用这里的值，否则保留 None，让 Agent 按模型名读取 model_config.json。
        self._base_url = base_url
        self._api_key = api_key
        if client is not None:
            self.client = client
        else:
            if base_url is not None or api_key is not None:
                planning_base_url = base_url
                planning_api_key = api_key
            else:
                planning_base_url, planning_api_key, found_model_config, missing_env_vars = _resolve_model_connection(model)
                if found_model_config and missing_env_vars:
                    missing_names = ", ".join(missing_env_vars)
                    raise ValueError(f"模型 {model} 配置存在，但缺少环境变量：{missing_names}")
                if not found_model_config:
                    planning_base_url = os.environ.get("OPENAI_BASE_URL")
                    planning_api_key = os.environ.get("OPENAI_API_KEY")
            self.client = OpenAI(base_url=planning_base_url, api_key=planning_api_key)
        self.mcp_client = mcp_client
        self.agent_runtime_config = agent_runtime_config
        self.agent_is_main_agent = agent_is_main_agent
        self.agent_class = agent_class

    def _resolve_agent_class(self):
        if self.agent_class is not None:
            return self.agent_class
        # 只有实际创建成员时才解析 Agent 类，multi_agent 可以独立被导入和测试。
        from agent_Teams import Agent

        return Agent

    def _create_agent(self, name: str, role: str):
        agent_class = self._resolve_agent_class()
        return agent_class(
            model=self.model,
            temperature=self.temperature,
            base_url=self._base_url,
            api_key=self._api_key,
            mcp_client=self.mcp_client,
            role=role,
            name=name,
            is_main_agent=self.agent_is_main_agent,
            runtime_config=self.agent_runtime_config,
        )

    def plan_team(self, task: str) -> list[dict[str, str]]:
        print("\n[PM] 分析任务，组建团队...")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": load_prompt("team_planning_system.md")},
                {"role": "user", "content": load_prompt("team_planning_user.md", task=task)}
            ],
            response_format={"type": "json_object"}
        )
        try:
            team = json.loads(response.choices[0].message.content).get("team", [])
            return team if isinstance(team, list) else []
        except Exception:
            return [{"name": "dev", "role": "developer", "task": task}]

    def run(self, task: str, members: list[dict[str, str]] | None = None):
        team = Team(agent_factory=self._create_agent)
        members = members or self.plan_team(task)

        print(f"\n[团队] {len(members)} 人")
        for i, member in enumerate(members, 1):
            print(f"  {i}. {member['name']} - {member['role']} -> {member['task']}")

        print(f"\n{'=' * 60}")
        print("  阶段 1: 招募团队")
        print(f"{'=' * 60}")
        for member in members:
            team.hire(member["name"], member["role"])

        print(f"\n{'=' * 60}")
        print("  阶段 2: 协作开始")
        print(f"{'=' * 60}")

        results = {}
        for i, member in enumerate(members):
            print(f"\n{'-' * 60}")
            print(f"  [{i + 1}/{len(members)}] {member['name']} 开始工作")
            print(f"{'-' * 60}")

            agent = team.agents[member["name"]]
            result = agent.chat(member["task"])
            results[member["name"]] = result
            team.broadcast(member["name"], f"我完成了任务。摘要：{str(result)[:200]}")

        if members:
            last = members[-1]
            reviewer = team.agents[last["name"]]

            print(f"\n{'=' * 60}")
            print(f"  阶段 3: {last['name']} 做最终审查")
            print(f"{'=' * 60}")

            review = reviewer.chat("请根据你收到的所有团队成果，做一个最终总结和审查。如果有问题请指出。")
            results["final_review"] = review

        print(f"\n{'=' * 60}")
        print("  阶段 4: 解散团队")
        print(f"{'=' * 60}")
        team.disband()

        print(f"\n{'=' * 60}")
        print("  最终成果")
        print(f"{'=' * 60}\n")
        for name, result in results.items():
            print(f"[{name}]")
            print(f"  {str(result)[:300]}\n")

        return results


def build_table_game_runtime_config(
        game_rule_file: str = "agent/multi_agent/阿瓦隆.txt",
        base_prompt_file: str = "agent/multi_agent/table_game_agent.md",
        extra_prompt_files: list[str] | None = None,
):
    from agent_Teams import AgentRuntimeConfig

    prompt_files = [game_rule_file]
    if extra_prompt_files:
        prompt_files.extend(extra_prompt_files)

    return AgentRuntimeConfig(
        # 桌游 Agent 只保留桌游提示词和规则，避免工程 rules、skills、memory、工具 schema 干扰推理。
        base_prompt_file=base_prompt_file,
        extra_prompt_files=prompt_files,
        include_rules=False,
        include_skills=False,
        include_memory=False,
        enable_local_tools=False,
        enable_mcp_tools=False,
    )


def create_table_game_orchestrator(**kwargs) -> TeamOrchestrator:
    runtime_config = kwargs.pop("agent_runtime_config", None) or build_table_game_runtime_config()
    return TeamOrchestrator(
        agent_runtime_config=runtime_config,
        agent_is_main_agent=False,
        **kwargs,
    )


def plan_team(task: str):
    return TeamOrchestrator().plan_team(task)


def run_team(task: str):
    return TeamOrchestrator().run(task)
