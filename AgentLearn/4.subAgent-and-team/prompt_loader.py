# encoding: utf-8
from pathlib import Path
from typing import Any


def load_prompt(name: str, prompts_dir: str | Path | None = None, **variables: Any) -> str:
    """
    根据prompt名称动态加载prompt内容
    """
    base_dir = Path(prompts_dir) if prompts_dir is not None else Path(__file__).parent / "agent" / "prompts"
    prompt_path = base_dir / name
    content = prompt_path.read_text(encoding="utf-8")
    for key, value in variables.items():
        content = content.replace(f"<{key}>", str(value))
    return content.strip()
