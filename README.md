# JanvisAgent

JanvisAgent,纯Python实现的一个类ClaudeCode的Agent。Janvis意为**just a not very intelligent system**，致敬钢铁侠的JARVIS，从0到1，实现一个功能完备的Agent，从最小 tool calling 开始，逐步加入长期记忆、MCP、skills、子 Agent、团队协作、会话管理和多Agent编排（多 Agent 游戏模拟为例）。

## 项目演进

| 阶段 | 目录 | 重点                                                                  |
| --- | --- |---------------------------------------------------------------------|
| 1 | `JanvisAgent/1.Hello` | 最小 Agent 循环，支持 bash、读文件、写文件三个最基础的工具调用。                              |
| 2 | `JanvisAgent/2.Memory` | 加入简单的长期记忆、任务规划和工具参数解析。                                              |
| 3 | `JanvisAgent/3.skills-and-mcp` | 引入各类rules、skills、本地实现的MCP客户端/服务端和系统提示词动态拼装。                         |
| 4 | `JanvisAgent/4.subAgent-and-team` | 构筑完全体，添加用户交互、流式响应、工具并发调度、优化的长期记忆、会话管理、子 Agent、团队协作和阿瓦隆游戏多 Agent 编排。 |

## 核心能力介绍
`4.subAgent-and-team` 阶段。

- `Agent_test.py`：推荐启动入口，会设置 Windows 终端 UTF-8，并自动连接或启动 TCP MCP 服务。
- `agent_Teams.py`：主 Agent 实现，负责交互循环、模型配置、工具调用、流式响应、上下文压缩、会话命令和任务完成检查。
    - Agent.run(): Agent loop入口
    - Agent.chat(task) Agent单次任务入口
- `tools/tool_manager.py`：统一管理本地工具和 MCP 工具，提供文件读写、检索、命令执行、后台命令、Web 搜索、子 Agent 调用等能力。
- `tools/tool_scheduler.py`：根据工具画像把只读且并发安全的工具并发执行，把有副作用的工具串行隔离。
- `session_manager.py`：把会话保存为追加式 JSONL，支持会话列表、加载、继续、标题、清空和按任务查看历史。
- `memory_manager.py`：后台整理长期记忆，保存完整任务上下文、任务摘要、主题记忆和 `MEMORY.md` 短索引。
- `mcp_client.py` / `mcp_server.py` / `mcp_tools.py`：自定义 JSON-RPC 风格 MCP 通信，目前实现了天气查询和携程机票低价趋势查询工具。
- `multi_agent.py`：团队编排实验，可根据任务规划成员、创建子 Agent、广播协作结果并做最终审查。
- `avalon_game.py`：阿瓦隆桌游多 Agent 编排实验，由程序主持一局游戏，Agent 玩家根据私有身份和公开日志行动。
- `module_test/`：模块测试目录，覆盖会话、记忆、工具管理、token 统计、流式响应、history 命令和完成度保护等逻辑。

## 启动方式

优先在环境变量中设置你的OPENAI兼容的base_url和api_key，然后修改Agent_test.py中传入的这两个参数。
或者直接传base_url和api_key的硬编码，需要注意信息安全风险。
（运行示例）推荐使用本机的 Anaconda Python：

```powershell
cd 项目目录

$env:LONGCAT_2_0_PREVIEW_BASE_URL="你的 OpenAI 兼容接口地址"
$env:LONGCAT_2_0_PREVIEW_API_KEY="你的 API Key"

python Agent_test.py
```

`Agent_test.py` 默认使用：

- 模型名：`LongCat-2.0-Preview`
- 环境变量：`LONGCAT_2_0_PREVIEW_BASE_URL`
- 环境变量：`LONGCAT_2_0_PREVIEW_API_KEY`
- MCP：TCP 模式，默认 `127.0.0.1:7777`，如果服务未启动会自动拉起本地 `mcp_server.py`

## 交互命令

启动第四阶段 Agent 后，可以直接输入自然语言任务，也可以输入内置命令：

| 命令 | 作用 |
| --- | --- |
| `help` / `h` | 查看帮助 |
| `tools` | 查看当前本地工具和 MCP 工具 |
| `status` | 查看当前模型、token 和运行状态 |
| `model_list` / `models` | 查看已配置模型 |
| `model add` | 新增模型配置并测试 |
| `model <name|编号>` | 切换模型 |
| `history` | 查看当前会话的用户任务历史，并可进入详情或删除任务 |
| `sessions` | 列出最近会话 |
| `continue` | 继续上一次非空会话 |
| `session current` | 查看当前会话信息 |
| `session new` | 创建新会话 |
| `session load <序号|session_id>` | 加载历史会话 |
| `session title <标题>` | 设置当前会话标题 |
| `compact` | 压缩当前短期上下文 |
| `memory compact` | 整理长期记忆索引 |
| `memory summarize sessions` | 扫描历史会话并提炼长期记忆 |
| `bash approve on/off` | 开启或关闭 bash 命令人工确认 |
| `clear current_session` | 清空当前会话，可选择同步删除相关记忆 |
| `clear sessions` | 选择删除历史会话及对应记忆 |
| `clear all_history` | 删除全部历史会话及对应记忆 |
| `exit` / `q` / `quit` | 退出 |

## 本地工具

本地工具定义在 `JanvisAgent/4.subAgent-and-team/tools/local_tools.json`，主要包括：

- 文件工具：`READ_FILE`、`WRITE_FILE`、`EDIT`
- 检索工具：`GLOB`、`GREP`、`LIST_DIR`
- 命令工具：`EXECUTE_BASH`
- 后台命令工具：`START_BACKGROUND_COMMAND`、`CHECK_BACKGROUND_COMMAND`、`READ_BACKGROUND_COMMAND_OUTPUT`、`STOP_BACKGROUND_COMMAND`
- 规划和扩展：`MAKE_PLAN`、`LOAD_SKILL_DETAIL_BY_NAME`
- 时间与联网：`GET_TIME`、`WEB_SEARCH`
- 记忆工具：`LOAD_FULL_MEMORY_CONTEXT`
- 子 Agent：`SUB_AGENT`

命令工具内置危险命令拦截，覆盖常见批量删除、格式化磁盘、管道下载脚本执行等高风险模式。

## 记忆与会话文件

运行第四阶段后会产生运行数据：

- `JanvisAgent/4.subAgent-and-team/sessions/`：按日期保存的会话 JSONL。
- `JanvisAgent/4.subAgent-and-team/agent/memory/`：长期记忆、任务索引、完整上下文和维护日志。
- `JanvisAgent/4.subAgent-and-team/runtime_output/`：命令工具默认输出目录。
- `JanvisAgent/4.subAgent-and-team/cache/`：例如城市三字码缓存。

这些文件偏运行态数据，修改代码或提交仓库时需要特别检查，避免把无关会话、缓存或敏感信息提交出去。

## 测试

第四阶段测试可以从阶段目录运行：
目录和python命令路径需要修改
```powershell
cd JanvisAgent\4.subAgent-and-team
python -m unittest discover -s module_test -p "test_*.py"
```

Web 搜索、MCP 天气和机票查询依赖外部网络，网络受限时相关测试或手动调用可能失败。

## 开发注意
- 当前版本只是具备较完整的雏形，还有可进一步开发的地方。
- 多数脚本依赖 OpenAI 兼容接口，优先通过环境变量传入 `base_url` 和 `api_key`。
- `local_tools.json` 是模型可见工具 schema，修改工具实现时要同步检查 schema、工具名常量和 `ToolManager` 映射。
- 历史会话、长期记忆和 cache 是运行数据，除非明确需要，否则尽量保存在本地。
