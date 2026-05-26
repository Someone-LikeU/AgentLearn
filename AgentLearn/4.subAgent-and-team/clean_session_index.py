# encoding: utf-8
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def read_jsonl_with_raw(path: Path) -> list[tuple[str, dict[str, Any] | None]]:
    """
    读取 JSONL，同时保留原始行，避免清理时破坏无法解析的历史内容。
    :param path: JSONL 文件路径
    :return: 原始行和解析结果
    """
    if not path.exists():
        return []
    rows: list[tuple[str, dict[str, Any] | None]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append((line, json.loads(line)))
        except json.JSONDecodeError:
            rows.append((line, None))
    return rows


def resolve_index_path(project_root: Path, index_path: str | None) -> Path:
    """
    解析 session_index.jsonl 路径。
    :param project_root: 当前阶段项目目录
    :param index_path: 用户传入的索引路径
    :return: 绝对路径
    """
    if index_path:
        path = Path(index_path)
        return path if path.is_absolute() else project_root / path
    return project_root / "sessions" / "session_index.jsonl"


def resolve_record_path(project_root: Path, raw_path: Any) -> Path | None:
    """
    解析索引记录中的 session 文件路径。
    :param project_root: 当前阶段项目目录
    :param raw_path: 索引记录 path 字段
    :return: 文件路径
    """
    if not raw_path:
        return None
    path = Path(str(raw_path))
    return path if path.is_absolute() else project_root / path


def session_file_exists(project_root: Path, sessions_dir: Path, session_id: str, records: list[dict[str, Any]]) -> bool:
    """
    判断索引中的 session 是否仍有真实文件。
    :param project_root: 当前阶段项目目录
    :param sessions_dir: sessions 目录
    :param session_id: 会话 id
    :param records: 同一 session_id 的索引记录
    :return: 是否存在
    """
    for record in records:
        path = resolve_record_path(project_root, record.get("path"))
        if path and path.exists():
            return True
    # 兼容旧索引缺 path 或 path 失效但文件仍按 session_id 存在的情况。
    return any(sessions_dir.rglob(f"{session_id}.jsonl"))


def build_cleaned_index(
    project_root: Path,
    sessions_dir: Path,
    rows: list[tuple[str, dict[str, Any] | None]],
) -> tuple[list[str], list[str]]:
    """
    生成清理后的索引行。
    :param project_root: 当前阶段项目目录
    :param sessions_dir: sessions 目录
    :param rows: 原始索引行
    :return: (保留行, 被移除的 session_id)
    """
    records_by_session: dict[str, list[dict[str, Any]]] = {}
    for _, record in rows:
        if not isinstance(record, dict):
            continue
        session_id = record.get("session_id")
        if session_id:
            records_by_session.setdefault(str(session_id), []).append(record)

    removed_session_ids = {
        session_id
        for session_id, records in records_by_session.items()
        if not session_file_exists(project_root, sessions_dir, session_id, records)
    }

    kept_lines: list[str] = []
    for raw_line, record in rows:
        if not isinstance(record, dict):
            # 无法解析的行不做推断删除，避免清理脚本扩大影响范围。
            kept_lines.append(raw_line)
            continue
        session_id = record.get("session_id")
        if session_id and str(session_id) in removed_session_ids:
            continue
        kept_lines.append(raw_line)
    return kept_lines, sorted(removed_session_ids)


def write_cleaned_index(index_file: Path, kept_lines: list[str], create_backup: bool) -> Path | None:
    """
    重写索引文件。
    :param index_file: session_index.jsonl 路径
    :param kept_lines: 保留的 JSONL 行
    :param create_backup: 是否创建备份
    :return: 备份路径
    """
    backup_path = None
    if create_backup and index_file.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = index_file.with_name(f"{index_file.name}.bak_{timestamp}")
        backup_path.write_text(index_file.read_text(encoding="utf-8"), encoding="utf-8")

    tmp_path = index_file.with_name(f"{index_file.name}.tmp")
    content = "\n".join(kept_lines)
    if content:
        content += "\n"
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(index_file)
    return backup_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean session_index.jsonl records whose session files no longer exist.")
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parent),
        help="4.subAgent-and-team directory. Default: this script directory.",
    )
    parser.add_argument("--index-file", default=None, help="Optional session_index.jsonl path.")
    parser.add_argument("--apply", action="store_true", help="Rewrite session_index.jsonl. Without it, only preview.")
    parser.add_argument("--no-backup", action="store_true", help="Do not create a .bak file when applying changes.")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    index_file = resolve_index_path(project_root, args.index_file)
    sessions_dir = index_file.parent

    rows = read_jsonl_with_raw(index_file)
    kept_lines, removed_session_ids = build_cleaned_index(project_root, sessions_dir, rows)

    print(f"index_file: {index_file}")
    print(f"total_lines: {len(rows)}")
    print(f"kept_lines: {len(kept_lines)}")
    print(f"removed_sessions: {len(removed_session_ids)}")
    for session_id in removed_session_ids:
        print(f"- {session_id}")

    if not args.apply:
        print("\ndry-run only. Use --apply to rewrite session_index.jsonl.")
        return 0

    backup_path = write_cleaned_index(index_file, kept_lines, create_backup=not args.no_backup)
    if backup_path:
        print(f"\nbackup: {backup_path}")
    print("session_index.jsonl cleaned.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
