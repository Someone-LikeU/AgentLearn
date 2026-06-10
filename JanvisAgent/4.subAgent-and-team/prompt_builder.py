# encoding: utf-8
# @Time    : 2026/04/24
import json
from typing import List
from prompt_loader import load_prompt


def build_system_prompt(base_prompt: List, rules, skills, memory=None):
	"""
	构造system prompt
	:param base_prompt: 最基本prompt，只包含角色设定
	:param rules: 设定
	:param skills: skills列表
	:param memory: 记忆已在rules模板中动态填充，这里保留参数兼容旧调用
	:return: system prompt
	"""
	# 拼接规则
	if rules:
		base_prompt.append(f"\n{rules}\n")
	# 拼接技能
	if skills:
		# 只有确实暴露 skill 时才拼接技能说明，轻量 Agent 可完全省掉这段 token。
		skill_prompt = None
		try:
			skill_prompt = load_prompt("skill_prompt_part.md")
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

	return "\n".join(base_prompt)
