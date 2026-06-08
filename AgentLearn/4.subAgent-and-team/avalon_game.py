# encoding: utf-8
import argparse
import io
import json
import random
import re
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_Teams import Agent, AgentRuntimeConfig
from rich.console import Console
from rich.text import Text


# 玩家池：玩家名 -> 使用的模型名。
PLAYER_POOL = {
    "龙猫": "LongCat-2.0-Preview",
    "阿辉": "LongCat-2.0-Preview",
    "二五仔": "LongCat-2.0-Preview",
    "小美": "LongCat-2.0-Preview",
    "阿珍": "glm-4.5-air",
    "老炮": "deepseek-r1:14b",
    "锤子": "glm-4.5-air",
    "大熊": "glm-4.5-air",
    "静香": "deepseek-r1:14b",
    "哆啦A梦": "deepseek-r1:14b",
}


ROLE_SIDE = {
    "梅林": "blue",
    "派西维尔": "blue",
    "忠臣": "blue",
    "刺客": "red",
    "莫甘娜": "red",
    "莫德雷德": "red",
    "奥伯伦": "red",
    "爪牙": "red",
}


ROLE_SETS = {
    5: ["梅林", "派西维尔", "忠臣", "刺客", "莫甘娜"],
    6: ["梅林", "派西维尔", "忠臣", "忠臣", "刺客", "莫甘娜"],
    7: ["梅林", "派西维尔", "忠臣", "忠臣", "刺客", "莫甘娜", "奥伯伦"],
    8: ["梅林", "派西维尔", "忠臣", "忠臣", "忠臣", "刺客", "莫甘娜", "莫德雷德"],
    9: ["梅林", "派西维尔", "忠臣", "忠臣", "忠臣", "忠臣", "刺客", "莫甘娜", "莫德雷德"],
    10: ["梅林", "派西维尔", "忠臣", "忠臣", "忠臣", "忠臣", "刺客", "莫甘娜", "莫德雷德", "奥伯伦"],
}


MISSION_TEAM_SIZES = {
    5: [2, 3, 2, 3, 3],
    6: [2, 3, 4, 3, 4],
    7: [2, 3, 3, 4, 4],
    8: [3, 4, 4, 5, 5],
    9: [3, 4, 4, 5, 5],
    10: [3, 4, 4, 5, 5],
}


RULE_FILE_CANDIDATES = [
    "agent/multi_agent/阿瓦隆.md"
]


@dataclass
class PlayerState:
    name: str
    model: str
    seat: int
    role: str
    side: str
    private_info: str
    agent: Agent


@dataclass
class AvalonGameState:
    players: dict[str, PlayerState]
    seat_order: list[str]
    play_order: list[str]
    direction: str
    leader_cursor: int = 0
    mission_index: int = 0
    success_count: int = 0
    fail_count: int = 0
    rejected_team_count: int = 0
    winner_side: str | None = None
    win_reason: str | None = None
    public_log: list[str] = field(default_factory=list)

    @property
    def player_names(self) -> list[str]:
        return list(self.players.keys())


class AvalonHost:
    def __init__(
            self,
            player_pool: dict[str, str],
            *,
            seed: int | None = None,
            player_count: int | None = None,
            direction: str | None = None,
            reveal_private_info: bool = True,
            max_vote_retries: int = 3,
            max_json_retries: int = 1,
            game_output_root: str | Path = "agent/multi_agent/game_records",
    ):
        self.player_pool = dict(player_pool)
        self.seed = seed
        self.player_count = player_count
        self.random = random.Random(seed)
        self.direction = direction
        self.reveal_private_info = reveal_private_info
        self.max_vote_retries = max_vote_retries
        self.max_json_retries = max(0, max_json_retries)
        self.state: AvalonGameState | None = None
        self.console = Console()
        self.game_output_root = self._resolve_output_root(game_output_root)
        self.game_output_dir: Path | None = None
        self._messages_dumped = False

    def run(self):
        self._setup_game()
        self._prepare_game_output_dir()
        try:
            self._host("游戏开始。")
            self._host(f"本局参赛玩家：{self._names_text(self.state.seat_order)}。")
            self._host(self._role_config_text(len(self.state.players)))
            self._host(self._mission_config_text(len(self.state.players)))
            self._print_seating()
            self._print_private_information()

            while not self._game_over_before_assassin():
                self._run_mission_round()

            if self.state.success_count >= 3:
                self._run_assassin_phase()
            elif self.state.fail_count >= 3:
                self._announce_red_mission_win()

            self._host("游戏结束。")
            self._print_final_roles()
        finally:
            try:
                self._dump_agent_messages()
            except Exception as error:
                self._host(f"保存玩家 self.messages 失败：{error}")

    def _setup_game(self):
        selected_player_pool = self._select_player_pool()
        player_count = len(selected_player_pool)
        if player_count not in ROLE_SETS:
            raise ValueError(f"当前支持 5-10 人局，收到 {player_count} 人。")

        names = list(selected_player_pool.keys())
        self.random.shuffle(names)
        roles = list(ROLE_SETS[player_count])
        self.random.shuffle(roles)

        direction = self.direction or self.random.choice(["clockwise", "counterclockwise"])
        play_order = names if direction == "clockwise" else list(reversed(names))
        players: dict[str, PlayerState] = {}

        for seat, (name, role) in enumerate(zip(names, roles), start=1):
            side = ROLE_SIDE[role]
            private_info = self._build_private_info(name, role, dict(zip(names, roles)))
            agent = self._create_player_agent(
                name,
                selected_player_pool[name],
                role,
                private_info,
                player_count,
                names,
                play_order,
                direction,
            )
            players[name] = PlayerState(
                name=name,
                model=selected_player_pool[name],
                seat=seat,
                role=role,
                side=side,
                private_info=private_info,
                agent=agent,
            )

        self.state = AvalonGameState(
            players=players,
            seat_order=names,
            play_order=play_order,
            direction=direction,
        )

    def _select_player_pool(self) -> dict[str, str]:
        if self.player_count is None:
            return dict(self.player_pool)
        if self.player_count < 1:
            raise ValueError("player_count 必须是正整数。")
        if self.player_count > len(self.player_pool):
            raise ValueError(f"玩家池只有 {len(self.player_pool)} 人，无法抽取 {self.player_count} 人。")

        # 先从玩家池随机抽取参赛玩家，再进入座次和身份随机流程。
        selected_names = self.random.sample(list(self.player_pool.keys()), self.player_count)
        return {name: self.player_pool[name] for name in selected_names}

    def _create_player_agent(
            self,
            name: str,
            model: str,
            role: str,
            private_info: str,
            player_count: int,
            seat_order: list[str],
            play_order: list[str],
            direction: str,
    ) -> Agent:
        runtime_config = AgentRuntimeConfig(
            base_prompt_file="agent/multi_agent/avalon_player_agent.md",
            prompt_variables={
                "game-rules": self._load_game_rules(),
                "player-count": player_count,
                "role-config": self._role_config_text(player_count),
                "mission-config": self._mission_config_text(player_count),
                "seat-order": self._seat_order_text_from(seat_order),
                "play-order": f"{self._names_text(play_order)}（{self._direction_text_from(direction)}）",
                "force-rule": self._force_rule_text(),
            },
            include_rules=False,
            include_skills=False,
            include_memory=False,
            enable_local_tools=False,
            enable_mcp_tools=False,
        )
        role_prompt = f"{role}（{self._side_text(ROLE_SIDE[role])}阵营）。{private_info}"
        # 初始化时屏蔽 Agent 自身加载提示，游戏过程由主持人统一打印。
        with redirect_stdout(io.StringIO()):
            return Agent(
                model=model,
                role=role_prompt,
                name=name,
                is_main_agent=False,
                runtime_config=runtime_config,
                temperature = 1.0
            )

    def _resolve_rule_file(self) -> str:
        for path in RULE_FILE_CANDIDATES:
            if Path(path).exists():
                return path
        raise FileNotFoundError("未找到阿瓦隆规则文件。")

    def _load_game_rules(self) -> str:
        # 每个玩家的 system prompt 通过占位符注入同一份规则，便于之后替换成其他桌游规则。
        return Path(self._resolve_rule_file()).read_text(encoding="utf-8").strip()

    def _role_config_text(self, player_count: int) -> str:
        role_counts = Counter(ROLE_SETS[player_count])
        parts = [
            f"{role}x{count}" if count > 1 else role
            for role, count in role_counts.items()
        ]
        return (
            f"{player_count} 人局，本局公开身份配置为：{self._names_text(parts)}。"
            "未出现在该配置中的角色不会出现在本局。"
        )

    def _mission_config_text(self, player_count: int) -> str:
        sizes = MISSION_TEAM_SIZES[player_count]
        size_text = "，".join(f"第{index + 1}轮{size}人" for index, size in enumerate(sizes))
        special = "7 人及以上第 4 轮任务需要 2 张失败票才失败。" if player_count >= 7 else "每轮任务 1 张失败票即失败。"
        return f"任务人数配置：{size_text}。{special}"

    def _build_private_info(self, name: str, role: str, role_by_name: dict[str, str]) -> str:
        side = ROLE_SIDE[role]
        red_players = [p for p, r in role_by_name.items() if ROLE_SIDE[r] == "red"]
        visible_red_to_evil = [
            p for p in red_players
            if p != name and role_by_name[p] != "奥伯伦" and role != "奥伯伦"
        ]
        merlin_visible = [
            p for p, r in role_by_name.items()
            if ROLE_SIDE[r] == "red" and r != "莫德雷德"
        ]
        percival_visible = [p for p, r in role_by_name.items() if r in ("梅林", "莫甘娜")]

        if role == "梅林":
            return f"你看到的红方玩家是：{self._names_text(merlin_visible)}。注意：莫德雷德不会出现在你的视野中。"
        if role == "派西维尔":
            return f"你看到的梅林候选人是：{self._names_text(percival_visible)}，其中一人可能是莫甘娜。"
        if side == "red":
            if role == "奥伯伦":
                return "你是红色阵营，但你当前不知道其他红方队友，其他红方队友当前也不知道你。"
            return f"你看到的红方队友是：{self._names_text(visible_red_to_evil)}。"
        return "你没有额外私密视野。"

    def _run_mission_round(self):
        assert self.state is not None
        mission_number = self.state.mission_index + 1
        mission_size = MISSION_TEAM_SIZES[len(self.state.players)][self.state.mission_index]
        fail_threshold = self._mission_fail_threshold()

        self._host(
            f"第 {mission_number} 轮任务开始。需要 {mission_size} 名队员，"
            f"失败票达到 {fail_threshold} 张则任务失败。"
        )

        while True:
            leader_name = self._current_leader()
            self._host(f"当前队长是 {leader_name}。")
            team = self._ask_leader_for_team(leader_name, mission_size)

            if self.state.rejected_team_count >= self.max_vote_retries:
                self._public(
                    f"已连续 {self.state.rejected_team_count} 次组队未通过，"
                    f"{leader_name} 本次组队强制执行，不再公开投票。"
                )
                self._advance_leader()
                self.state.rejected_team_count = 0
                self._run_mission(team)
                self.state.mission_index += 1
                return

            approved = self._run_team_vote(leader_name, team)
            self._advance_leader()

            if approved:
                self.state.rejected_team_count = 0
                self._run_mission(team)
                self.state.mission_index += 1
                return

            self.state.rejected_team_count += 1
            self._public(f"组队未通过。连续组队失败次数：{self.state.rejected_team_count}。")

    def _ask_leader_for_team(self, leader_name: str, mission_size: int) -> list[str]:
        prompt = self._build_action_prompt(
            leader_name,
            action="组队",
            instruction=(
                f"你是当前队长。请提名 {mission_size} 名任务队员，可以包含你自己。\n"
                '只输出 JSON：{"speech":"你的公开发言","team":["玩家名1","玩家名2"]}'
            ),
        )
        data = self._ask_agent_json(leader_name, prompt)
        speech = self._clean_text(data.get("speech") or "我先给出一个我认为稳定的队伍。")
        team = self._normalize_team(data.get("team"), mission_size, leader_name)
        self._player_speech(leader_name, speech)
        self._public(f"{leader_name} 提名队伍：{self._names_text(team)}。")
        return team

    def _run_team_vote(self, leader_name: str, team: list[str]) -> bool:
        assert self.state is not None
        votes: dict[str, str] = {}
        speeches: dict[str, str] = {}

        self._host(f"所有玩家对队伍 {self._names_text(team)} 投票。")
        for name in self.state.play_order:
            prompt = self._build_action_prompt(
                name,
                action="投票",
                instruction=(
                    f"队长 {leader_name} 提名的队伍是：{self._names_text(team)}。\n"
                    '请投票。只输出 JSON：{"speech":"你的简短投票理由","vote":"approve 或 reject"}'
                ),
            )
            data = self._ask_agent_json(name, prompt)
            vote = str(data.get("vote") or "").strip().lower()
            if vote not in {"approve", "reject"}:
                vote = "reject"
            votes[name] = vote
            speeches[name] = self._clean_text(data.get("speech") or "我先按当前信息投票。")

        approve_count = sum(1 for vote in votes.values() if vote == "approve")
        reject_count = len(votes) - approve_count
        approved = approve_count > reject_count
        result_text = "通过" if approved else "未通过"

        vote_details = []
        for name in self.state.play_order:
            self._player_speech(name, speeches[name], record=False, broadcast=False)
            vote_text = self._vote_text(votes[name])
            self._host(f"{name} 投票：{vote_text}")
            vote_details.append(f"{name}：{vote_text}，理由：{speeches[name]}")

        # 所有玩家同时投票；主持人收齐后再统一公开票型和发车结果。
        self._public(
            f"投票公开：{self._names_text(vote_details)}。"
            f"投票结果：赞成 {approve_count}，反对 {reject_count}，队伍{result_text}。"
        )
        return approved

    def _run_mission(self, team: list[str]):
        assert self.state is not None
        actions: dict[str, str] = {}
        self._host(f"任务队员 {self._names_text(team)} 秘密提交任务牌。")

        for name in team:
            prompt = self._build_action_prompt(
                name,
                action="任务",
                instruction=(
                    f"你在任务队伍中。你的阵营是{self._side_text(self.state.players[name].side)}。\n"
                    '请秘密选择任务牌。只输出 JSON：{"speech":"一句不会公开的内心策略","action":"success 或 fail"}'
                ),
            )
            data = self._ask_agent_json(name, prompt)
            action = str(data.get("action") or "").strip().lower()
            action = self._normalize_mission_action(name, action)
            actions[name] = action
            self._host(f"{name} 已提交任务牌。")

        fail_votes = sum(1 for action in actions.values() if action == "fail")
        fail_threshold = self._mission_fail_threshold()
        if fail_votes >= fail_threshold:
            self.state.fail_count += 1
            result_text = "失败"
        else:
            self.state.success_count += 1
            result_text = "成功"

        self._public(
            f"第 {self.state.mission_index + 1} 轮任务{result_text}。"
            f"失败票数：{fail_votes}。当前比分：蓝方成功 {self.state.success_count}，红方破坏 {self.state.fail_count}。"
        )

    def _run_assassin_phase(self):
        assert self.state is not None
        assassin_name = self._find_role("刺客")
        merlin_name = self._find_role("梅林")
        if not assassin_name or not merlin_name:
            self._set_winner("blue", "缺少刺客或梅林身份，蓝方 3 次任务成功后无法完成刺杀。")
            self._public("缺少刺客或梅林身份，跳过刺杀阶段。蓝色阵营获胜。")
            return

        self._host("蓝方已完成 3 次任务，进入刺客刺杀阶段。")
        prompt = self._build_action_prompt(
            assassin_name,
            action="刺杀",
            instruction=(
                "你是刺客。请根据全局发言和任务结果选择你认为的梅林。\n"
                '只输出 JSON：{"speech":"你的刺杀理由","target":"玩家名"}'
            ),
        )
        data = self._ask_agent_json(assassin_name, prompt)
        target = str(data.get("target") or "").strip()
        if target not in self.state.players or target == assassin_name:
            target = self._fallback_assassin_target(assassin_name)
        speech = self._clean_text(data.get("speech") or "我根据发言和投票选择刺杀目标。")
        self._player_speech(assassin_name, speech)
        self._public(f"刺客 {assassin_name} 选择刺杀 {target}。")

        if target == merlin_name:
            self._set_winner("red", f"刺客 {assassin_name} 成功刺杀梅林 {target}。")
            self._public(f"刺杀成功，{target} 是梅林。红色阵营获胜。")
        else:
            self._set_winner("blue", f"刺客 {assassin_name} 刺杀 {target} 失败，梅林是 {merlin_name}。")
            self._public(f"刺杀失败，梅林是 {merlin_name}。蓝色阵营获胜。")

    def _announce_red_mission_win(self):
        assert self.state is not None
        # 红方达到 3 次任务破坏时不进入刺杀阶段，直接按任务比分结算。
        reason = (
            f"当前比分：蓝方成功 {self.state.success_count}，"
            f"红方破坏 {self.state.fail_count}。红色阵营完成 3 次任务破坏。"
        )
        self._set_winner("red", reason)
        self._public(f"最终结果：{reason}红色阵营获胜。")

    def _set_winner(self, side: str, reason: str):
        assert self.state is not None
        self.state.winner_side = side
        self.state.win_reason = reason

    def _ask_agent_json(self, player_name: str, prompt: str) -> dict[str, Any]:
        assert self.state is not None
        agent = self.state.players[player_name].agent
        action = self._prompt_action(prompt)
        expected_keys = self._expected_json_keys(action)
        current_prompt = prompt
        last_raw_text = ""

        for attempt in range(self.max_json_retries + 1):
            try:
                with redirect_stdout(io.StringIO()) as output:
                    raw = agent.chat(current_prompt)
                last_raw_text = str(raw or output.getvalue()).strip()
            except Exception as error:
                self._host(f"{player_name} 调用 Agent 失败，使用主持人兜底动作。错误：{error}")
                return {}

            data = self._parse_json_object(last_raw_text, expected_keys=expected_keys)
            if data is not None:
                return data

            reason = "输出为空" if not last_raw_text else "输出无法解析为 JSON"
            if attempt < self.max_json_retries:
                self._host(f"{player_name} {reason}，要求重新只输出 JSON。原始输出：{last_raw_text[:200]}")
                current_prompt = self._build_json_retry_prompt(action, last_raw_text)
                continue

        self._host(f"{player_name} 输出无法解析为 JSON，使用主持人兜底动作。原始输出：{last_raw_text[:200]}")
        return {}

    def _build_json_retry_prompt(self, action: str | None, raw_text: str) -> str:
        expected_json = self._expected_json_example(action)
        return (
            "【主持人要求重新作答】\n"
            "你上一次回复为空或不是可解析的 JSON。请根据上一条主持人指令重新思考并作答。\n"
            f"必须只输出一个 JSON 对象，格式为：{expected_json}\n"
            "不要输出 Markdown、代码块、解释、思考过程、</think> 或多余文字。\n"
            f"上一条无效输出：{raw_text[:500]}"
        )

    def _prompt_action(self, prompt: str) -> str | None:
        match = re.search(r"【主持人指令：([^】]+)】", prompt or "")
        return match.group(1) if match else None

    def _expected_json_keys(self, action: str | None) -> set[str] | None:
        if action == "组队":
            return {"speech", "team"}
        if action == "投票":
            return {"speech", "vote"}
        if action == "任务":
            return {"speech", "action"}
        if action == "刺杀":
            return {"speech", "target"}
        return None

    def _expected_json_example(self, action: str | None) -> str:
        if action == "组队":
            return '{"speech":"你的公开发言","team":["玩家名1","玩家名2"]}'
        if action == "投票":
            return '{"speech":"你的简短投票理由","vote":"approve 或 reject"}'
        if action == "任务":
            return '{"speech":"一句不会公开的内心策略","action":"success 或 fail"}'
        if action == "刺杀":
            return '{"speech":"你的刺杀理由","target":"玩家名"}'
        return '{"speech":"你的回答"}'

    def _build_action_prompt(self, player_name: str, *, action: str, instruction: str) -> str:
        assert self.state is not None
        visible_state = (
            f"当前任务轮次：{self.state.mission_index + 1}\n"
            f"当前比分：蓝方成功 {self.state.success_count}，红方破坏 {self.state.fail_count}\n"
            f"连续组队失败次数：{self.state.rejected_team_count}\n"
            "固定局面和已公开事件已在你的 system prompt 与此前主持人广播中给出。"
        )
        return (
            f"【主持人指令：{action}】\n"
            f"你是 {player_name}。\n\n"
            f"{visible_state}\n\n"
            f"{instruction}"
        )

    def _normalize_team(self, raw_team: Any, mission_size: int, leader_name: str) -> list[str]:
        assert self.state is not None
        team: list[str] = []
        if isinstance(raw_team, list):
            for item in raw_team:
                name = str(item).strip()
                if name in self.state.players and name not in team:
                    team.append(name)

        # 队伍人数不足或输出非法时，主持人按行动顺序从队长开始补齐，保证流程继续。
        ordered = self._ordered_from_leader(leader_name)
        for name in ordered:
            if len(team) >= mission_size:
                break
            if name not in team:
                team.append(name)
        return team[:mission_size]

    def _normalize_mission_action(self, player_name: str, action: str) -> str:
        assert self.state is not None
        player = self.state.players[player_name]
        if player.side == "blue":
            return "success"
        if action not in {"success", "fail"}:
            return "fail"
        return action

    def _current_leader(self) -> str:
        assert self.state is not None
        return self.state.play_order[self.state.leader_cursor % len(self.state.play_order)]

    def _advance_leader(self):
        assert self.state is not None
        self.state.leader_cursor = (self.state.leader_cursor + 1) % len(self.state.play_order)

    def _ordered_from_leader(self, leader_name: str) -> list[str]:
        assert self.state is not None
        order = self.state.play_order
        start = order.index(leader_name)
        return order[start:] + order[:start]

    def _mission_fail_threshold(self) -> int:
        assert self.state is not None
        player_count = len(self.state.players)
        mission_number = self.state.mission_index + 1
        if player_count >= 7 and mission_number == 4:
            return 2
        return 1

    def _game_over_before_assassin(self) -> bool:
        assert self.state is not None
        return self.state.success_count >= 3 or self.state.fail_count >= 3

    def _find_role(self, role: str) -> str | None:
        assert self.state is not None
        for player in self.state.players.values():
            if player.role == role:
                return player.name
        return None

    def _resolve_output_root(self, output_root: str | Path) -> Path:
        output_path = Path(output_root)
        if output_path.is_absolute():
            return output_path
        return Path(__file__).resolve().parent / output_path

    def _prepare_game_output_dir(self):
        assert self.state is not None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        seed_part = f"_seed_{self.seed}" if self.seed is not None else ""
        dirname = f"avalon_{timestamp}_{len(self.state.players)}p{seed_part}"
        self.game_output_dir = self.game_output_root / dirname
        # 每局游戏单独建目录，避免多局复盘文件互相覆盖。
        self.game_output_dir.mkdir(parents=True, exist_ok=False)
        self._host(f"本局复盘目录：{self.game_output_dir}")

    def _dump_agent_messages(self):
        if self._messages_dumped or self.state is None or self.game_output_dir is None:
            return
        self._messages_dumped = True
        messages_dir = self.game_output_dir / "agent_messages"
        messages_dir.mkdir(parents=True, exist_ok=True)

        for player in self.state.players.values():
            filename = f"{player.seat:02d}_{self._safe_filename(player.name)}_{self._safe_filename(player.role)}.json"
            output_path = messages_dir / filename
            payload = {
                "player": player.name,
                "model": player.model,
                "seat": player.seat,
                "role": player.role,
                "side": player.side,
                "messages": self._json_safe(player.agent.messages),
            }
            output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        summary = {
            "player_count": len(self.state.players),
            "seed": self.seed,
            "direction": self.state.direction,
            "seat_order": self.state.seat_order,
            "play_order": self.state.play_order,
            "score": {
                "blue_success": self.state.success_count,
                "red_fail": self.state.fail_count,
            },
            "winner": {
                "side": self.state.winner_side,
                "side_text": self._side_text(self.state.winner_side) if self.state.winner_side else None,
                "reason": self.state.win_reason,
            },
            "players": [
                {
                    "name": player.name,
                    "model": player.model,
                    "seat": player.seat,
                    "role": player.role,
                    "side": player.side,
                }
                for player in sorted(self.state.players.values(), key=lambda item: item.seat)
            ],
            "public_log": self.state.public_log,
        }
        (self.game_output_dir / "game_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._host(f"已保存所有玩家 self.messages：{messages_dir}")

    def _safe_filename(self, value: str) -> str:
        safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(value or "").strip())
        return safe or "unknown"

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [self._json_safe(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if hasattr(value, "model_dump"):
            return self._json_safe(value.model_dump())
        if hasattr(value, "to_dict"):
            return self._json_safe(value.to_dict())
        return str(value)

    def _fallback_assassin_target(self, assassin_name: str) -> str:
        assert self.state is not None
        candidates = [name for name in self.state.player_names if name != assassin_name]
        return self.random.choice(candidates)

    def _parse_json_object(self, text: str, expected_keys: set[str] | None = None) -> dict[str, Any] | None:
        text = str(text or "").strip()
        if not text:
            return None
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass

        decoder = json.JSONDecoder()
        candidates: list[dict[str, Any]] = []
        for match in re.finditer(r"\{", text):
            try:
                data, _ = decoder.raw_decode(text[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                candidates.append(data)

        if not candidates:
            return None
        if expected_keys:
            for candidate in candidates:
                if expected_keys.issubset(candidate.keys()):
                    return candidate
        return candidates[0]

    def _print_seating(self):
        assert self.state is not None
        self._host(f"随机座次：{self._seat_order_text()}。")
        self._host(f"本局行动方向：{self._direction_text()}。行动顺序：{self._names_text(self.state.play_order)}。")

    def _print_private_information(self):
        assert self.state is not None
        for name in self.state.seat_order:
            player = self.state.players[name]
            if self.reveal_private_info:
                self._host(f"私信 {name}：身份 {player.role}，{player.private_info}")
            else:
                self._host(f"已向 {name} 发送身份和私密视野。")

    def _print_final_roles(self):
        assert self.state is not None
        self._host("最终身份公开：")
        for name in self.state.seat_order:
            player = self.state.players[name]
            style = self._side_style(player.side)
            self.console.print(Text(f"  {name}：{player.role}（{self._side_text(player.side)}阵营，模型 {player.model}）", style=style))

    def _public(self, message: str):
        assert self.state is not None
        self._host(message)
        self._record_public_event(message)
        self._broadcast_public_message(message)

    def _host(self, message: str):
        self.console.print()
        self.console.print(Text(f"[主持人] {message}", style="green"))

    def _player_speech(self, player_name: str, speech: str, *, record: bool = True, broadcast: bool = True):
        style = "white"
        if self.state is not None and player_name in self.state.players:
            style = self._side_style(self.state.players[player_name].side)
        self.console.print(Text(f"[{player_name} 发言] {speech}", style=style))
        if self.state is not None and record:
            message = f"{player_name} 发言：{speech}"
            self._record_public_event(message)
            if broadcast:
                self._broadcast_public_message(message, exclude_names={player_name})

    def _record_public_event(self, message: str):
        assert self.state is not None
        self.state.public_log.append(message)

    def _broadcast_public_message(self, message: str, exclude_names: set[str] | None = None):
        assert self.state is not None
        excluded = exclude_names or set()
        for name, player in self.state.players.items():
            if name in excluded:
                continue
            # 公开信息只追加一次到每名玩家的短期上下文，避免后续动作提示重复携带完整公开历史。
            player.agent.messages.append({
                "role": "user",
                "content": f"【主持人广播】\n{message}",
            })

    def _recent_public_log(self, limit: int = 30) -> str:
        assert self.state is not None
        if not self.state.public_log:
            return "暂无公开历史。"
        return "\n".join(f"- {item}" for item in self.state.public_log[-limit:])

    def _seat_order_text(self) -> str:
        assert self.state is not None
        return self._seat_order_text_from(self.state.seat_order)

    def _seat_order_text_from(self, seat_order: list[str]) -> str:
        return "，".join(f"{idx + 1}号位 {name}" for idx, name in enumerate(seat_order))

    def _direction_text(self) -> str:
        assert self.state is not None
        return self._direction_text_from(self.state.direction)

    def _direction_text_from(self, direction: str) -> str:
        return "顺时针" if direction == "clockwise" else "逆时针"

    def _force_rule_text(self) -> str:
        return f"连续 {self.max_vote_retries} 次组队未通过后，下一任队长组队将直接执行任务，不再公开投票。"

    def _names_text(self, names: list[str]) -> str:
        return "、".join(names) if names else "无"

    def _side_text(self, side: str) -> str:
        return "蓝色" if side == "blue" else "红色"

    def _side_style(self, side: str) -> str:
        return "blue" if side == "blue" else "red"

    def _vote_text(self, vote: str) -> str:
        return "赞成" if vote == "approve" else "反对"

    def _clean_text(self, value: Any) -> str:
        text = str(value or "").strip()
        return text if text else "我先保留意见。"


def run_game(
        player_pool: dict[str, str] | None = None,
        seed: int | None = None,
        player_count: int | None = None,
        game_output_root: str | Path = "agent/multi_agent/game_records",
        max_vote_retries: int = 3,
        max_json_retries: int = 1,
):
    host = AvalonHost(
        player_pool or PLAYER_POOL,
        seed=seed,
        player_count=player_count,
        game_output_root=game_output_root,
        max_vote_retries=max_vote_retries,
        max_json_retries=max_json_retries,
    )
    host.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行一局由程序主持、Agent 玩家参与的阿瓦隆游戏。")
    parser.add_argument("-n", "--player-count", type=int, default=None, help="从玩家池随机抽取的玩家人数。")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，便于复现实验。")
    parser.add_argument(
        "--game-output-root",
        default="agent/multi_agent/game_records",
        help="每局游戏复盘文件的输出根目录。",
    )
    parser.add_argument("--max-vote-retries", type=int, default=3, help="连续几次组队未通过后，下一次组队强制执行。")
    parser.add_argument("--max-json-retries", type=int, default=1, help="模型输出为空或非 JSON 时的重新作答次数。")
    args = parser.parse_args()
    run_game(
        seed=args.seed,
        player_count=args.player_count,
        game_output_root=args.game_output_root,
        max_vote_retries=args.max_vote_retries,
        max_json_retries=args.max_json_retries,
    )
