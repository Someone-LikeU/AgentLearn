Judge whether the active task is complete.

Return only one word: DONE, CONTINUE, NEED_USER, or BLOCKED.

Rules:
- Judge only the active task, not the parent task or the full plan.
- DONE means the active task has been completed.
- CONTINUE means more autonomous work is needed and can be done without user input.
- NEED_USER means progress requires more information or confirmation from the user.
- BLOCKED means the task cannot continue because of errors, unavailable tools, or other blockers.
