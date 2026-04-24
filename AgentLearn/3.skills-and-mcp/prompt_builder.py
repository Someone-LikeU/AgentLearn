# encoding: utf-8
# @Time    : 2026/04/24
from typing import List


def build_system_prompt(base_prompt: List, rules, skills, memory):
	"""
	构造system prompt
	:param base_prompt: 最基本prompt，只包含角色设定
	:param rules: 设定
	:param skills: skills
	:param memory: 记忆
	:return: system prompt
	"""
	# 拼接规则
	if rules:
		base_prompt.append(f"\n{rules}\n")
	# 拼接技能
	if skills:
		skill_prompt = None
		try:
			with open("./agent/SKILL_PROMPT_PART.md", 'r', encoding='utf-8') as f:
				skill_prompt = f.read()
			"""
				TODO 构造成
				 skill_prompt
				 <available_skills>
				 [{"name": "", "description": ""}]
				 </available_skills>
				 的样子
			"""
			skill_prompt = f"{skill_prompt}\n{"\n".join([f"- {skill['name']}: {skill.get('description', '')}" for skill in skills])}"
			skill_prompt = None
		except FileNotFoundError as e:
			pass

