# encoding: utf-8
# @Time    : 2026/04/24
import json
from typing import List


def build_system_prompt(base_prompt: List, rules, skills, memory):
	"""
	构造system prompt
	:param base_prompt: 最基本prompt，只包含角色设定
	:param rules: 设定
	:param skills: skills列表
	:param memory: 记忆
	:return: system prompt
	"""
	# 拼接规则
	if rules:
		base_prompt.append(f"\n{rules}\n")
	# 拼接技能
	skill_prompt = None
	try:
		with open("./agent/SKILL_PROMPT_PART.md", 'r', encoding='utf-8') as f:
			skill_prompt = f.read()
		available_skills = {
			"available_skills": [
				{"name": skill["name"], "description": skill.get("description", "")}
				for skill in skills
			]
		}
		skills_json = json.dumps(available_skills, ensure_ascii=False, indent=2)
		skill_prompt = f"\n{skill_prompt}\n```JSON\n{skills_json}\n```"
		base_prompt.append(skill_prompt)
	except FileNotFoundError as e:
		print("Error: File not found ", e)

	# 拼接记忆
	if memory:
		base_prompt.append(f"\n# Previous context\n{memory}")

	return "\n".join(base_prompt)
