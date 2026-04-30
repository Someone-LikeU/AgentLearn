# 本地语义搜索技能 - 使用指南

## 概述

本地语义搜索技能是一个运行在个人电脑上的智能文件搜索工具，支持：
- 语义理解搜索（非关键词匹配）
- 多格式文档支持（txt, md, pdf, docx, pptx）
- 本地向量数据库索引
- 中文/英文双语支持
- 目录索引状态检查
- 全量/增量刷新选择
- CLI自然语言触发
- 本地模型灵活切换
- 模型加载失败自动提示解决方案

**版本：v1.5.0**

---

## 第一章：CLI自然语言操作

在 CodeAgent CLI 中，可以直接使用自然语言触发 skill 的各种操作。

### 1.1 搜索操作

**触发词：**
- "本地搜索..."
- "帮我搜索..."
- "在本地找..."
- "查找文件..."
- "在某个目录搜索..."

**示例：**

| 自然语言请求 | 执行操作 |
|--------------|----------|
| 帮我在本地搜索年结应急预案 | 检查索引状态 → 执行搜索 → 返回结果 |
| 本地搜索BCM调度方案 | 使用当前索引执行搜索 |
| 在本地找关于报告的文件 | 语义搜索，返回相关文档 |
| 帮我在 D:\文档 目录中搜索数据分析 | 检查目录索引 → 提示刷新选项 → 搜索 |

**指定目录搜索流程：**

当用户说"帮我在某个目录搜索..."时：

1. 系统自动检查该目录索引状态
2. 显示索引信息：
   ```
   目录索引状态：
   - 已索引文件数：300/465
   - 索引时间：2026-04-28
   - 状态：部分完成
   ```
3. 提示用户选择：
   ```
   请选择刷新方式：
   1. 全量刷新（重新索引所有文件）
   2. 增量刷新（只更新变化的文件）
   3. 不刷新（使用现有索引搜索）
   ```
4. 用户选择后执行相应操作
5. 执行搜索并返回结果

### 1.2 索引刷新操作

**触发词：**
- "刷新索引..."
- "更新索引..."
- "重建索引..."
- "帮我刷新某个目录的索引..."

**示例：**

| 自然语言请求 | 执行操作 |
|--------------|----------|
| 帮我刷新 D:\文档 目录的索引 | 检查状态 → 提示选择 → 执行刷新 |
| 更新本地搜索索引 | 增量刷新当前索引目录 |
| 重建索引 | 全量刷新当前索引目录 |
| 刷新 LTS 目录索引 | 全量刷新指定目录 |

**刷新方式选择：**

当用户请求刷新索引时，系统会提示：

```
请选择刷新方式：
1. 全量刷新（删除旧索引，重新构建所有文件，耗时较长）
2. 增量刷新（检测变更文件，只更新变化的部分，速度快）

请输入选择（1或2）：
```

### 1.3 索引状态检查

**触发词：**
- "检查索引状态..."
- "查看索引情况..."
- "索引是否过期..."

**示例：**

| 自然语言请求 | 执行操作 |
|--------------|----------|
| 检查 D:\文档 目录的索引状态 | 显示索引详细信息 |
| 查看当前索引情况 | 显示已索引目录和文件数 |
| 索引是否过期 | 检测文件变更，提示是否需要刷新 |

### 1.4 模型管理

**触发词：**
- "切换模型..."
- "扫描本地模型..."
- "查看可用模型..."
- "下载新模型..."

**示例：**

| 自然语言请求 | 执行操作 |
|--------------|----------|
| 帮我扫描本地模型 | 显示本地缓存的嵌入模型列表 |
| 切换到其他模型 | 交互式选择本地模型 |
| 查看支持的模型 | 列出推荐的嵌入模型 |
| 下载新模型 | 下载指定的嵌入模型 |

**模型加载失败时的提示：**

当模型加载失败时，系统会自动提示：

```
模型加载失败: [错误信息]

解决方案:
1. 检查本地模型缓存:
   python scripts/model_manager.py --scan
2. 交互式选择本地模型:
   python scripts/model_manager.py --interactive
3. 下载新模型:
   python scripts/model_manager.py --download moka-ai/m3e-base
4. 切换到其他模型:
   python scripts/model_manager.py --switch BAAI/bge-large-zh
```

### 1.5 重置向量库

**触发词：**
- "重置向量库..."
- "清除向量库..."
- "清空索引..."
- "恢复初始状态..."

**示例：**

| 自然语言请求 | 执行操作 |
|--------------|----------|
| 帮我重置向量库 | 显示状态 → 确认提示 → 执行重置 |
| 清除本地搜索索引 | 删除所有向量数据 |
| 清空向量库恢复初始状态 | 重置为初始状态 |
| 重置索引数据 | 删除向量库所有数据 |

**重置流程：**

当用户请求重置向量库时：

```
系统: 正在查看向量库状态...

向量库状态：
- 总大小：4.04 MB
- 已索引文件数：300
- 索引目录：D:\文档\05.模块服务\07.LTS

警告：重置操作将删除所有向量库数据，不可恢复！

请确认是否继续：
1. 确认重置
2. 取消操作
```

用户确认后执行重置，恢复为初始状态。

**适用场景：**
- 索引损坏需要重建
- 切换嵌入模型后需要清空旧数据
- 删除所有索引重新开始
- 磁盘空间不足需要清理
| 索引是否过期 | 检测文件变更，提示是否需要刷新 |

### 1.4 完整对话示例

**场景1：首次搜索新目录**

```
用户: 帮我在 D:\文档\05.模块服务\07.LTS 目录中搜索年结应急预案

系统: 正在检查目录索引状态...

目录索引状态：
- 目录：D:\文档\05.模块服务\07.LTS
- 状态：未索引
- 文件数：465

该目录尚未建立索引，需要构建索引才能搜索。

请选择：
1. 全量构建索引（约5分钟）
2. 暂不构建，使用其他已索引目录

用户: 1

系统: 开始构建索引...
已索引：bcm_func-pjob_sch_filter_modify_pkg.txt
已索引：LTS入湖数据地址.txt
...
索引构建完成，共465个文件

正在搜索：年结应急预案
找到10个结果：
[1] 2024年结保障-数据调度服务.docx (相似度81%)
[2] 财经年结保障应预案.docx (相似度80%)
...
```

**场景2：搜索已索引目录**

```
用户: 帮我在本地搜索BCM调度方案

系统: 使用当前索引目录：D:\文档\05.模块服务\07.LTS
正在搜索：BCM调度方案

找到10个结果：
[1] LTS 一天多调方案.pptx (相似度84%)
[2] LTS支撑BCM 1.0切换一天多调方案.pptx (相似度83%)
...
```

**场景3：更新索引**

```
用户: 帮我刷新 LTS 目录的索引

系统: 正在检查目录索引状态...

目录索引状态：
- 已索引文件数：300/465
- 索引时间：2026-04-28
- 状态：部分完成

请选择刷新方式：
1. 全量刷新（重新索引所有文件）
2. 增量刷新（只更新变化的文件）

用户: 2

系统: 增量更新模式...
新增文件：5
更新文件：3
删除文件：2
增量更新完成！
```

**场景4：重置向量库**

```
用户: 帮我重置向量库

系统: 正在查看向量库状态...

向量库状态：
- 总大小：4.04 MB
- 已索引文件数：300
- 索引目录：D:\文档\05.模块服务\07.LTS

警告：重置操作将删除所有向量库数据，不可恢复！

请确认是否继续：
1. 确认重置
2. 取消操作

用户: 1

系统: 正在重置向量库...
删除文件：chroma.sqlite3
删除文件：metadata.json
删除目录：...

向量库已重置，恢复为初始状态！
可以重新构建索引。
```

---

## 第二章：命令行操作

对于高级用户或自动化场景，可以直接使用命令行脚本。

### 2.1 安装依赖

```bash
cd ~/.config/opencode/skills/local-semantic-search
pip install -r requirements.txt
```

依赖包：
- chromadb（向量数据库）
- sentence-transformers（嵌入模型）
- PyPDF2（PDF解析）
- python-docx（Word解析）
- python-pptx（PPT解析）

### 2.2 索引状态检查

**命令：**
```bash
python scripts/check_index.py --dir <目录路径> [--json]
```

**参数：**
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dir` | 要检查的目录路径 | 必填 |
| `--json` | 输出JSON格式 | False |

**示例：**
```bash
# 文本输出
python scripts/check_index.py --dir "D:/文档/05.模块服务/07.LTS"

# JSON输出
python scripts/check_index.py --dir "D:/文档" --json
```

**输出示例：**
```
目录: D:\文档\05.模块服务\07.LTS
状态: 索引部分完成
目录文件数: 465
已索引文件数: 300
索引时间: 2026-04-28T15:00:00
向量数: 300
文件类型: txt, md, pdf, docx, pptx
```

**状态说明：**
| 状态 | 说明 | 建议 |
|------|------|------|
| `not_indexed` | 目录未索引 | 需要构建索引 |
| `complete` | 索引完整 | 可直接搜索 |
| `partial` | 索引部分完成 | 建议全量刷新 |
| `stale` | 索引可能过期 | 建议增量刷新 |
| `different_dir` | 当前索引其他目录 | 需要重新索引 |

### 2.3 索引构建与刷新

**全量刷新命令：**
```bash
python scripts/indexer.py --dir <目录> --force
```

**增量刷新命令：**
```bash
python scripts/indexer.py --dir <目录> --incremental
```

**参数：**
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dir` | 搜索根目录 | 当前目录 |
| `--include-content` | 索引文件内容 | True |
| `--file-types` | 文件类型过滤 | txt,md,pdf,docx,pptx |
| `--force` | 全量刷新 | False |
| `--incremental` | 增量刷新 | False |

**示例：**
```bash
# 全量刷新
python scripts/indexer.py --dir "D:/文档" --force

# 增量刷新
python scripts/indexer.py --dir "D:/文档" --incremental

# 指定文件类型
python scripts/indexer.py --dir "D:/文档" --file-types txt,md,pptx --force

# 不包含内容（只索引文件名）
python scripts/indexer.py --dir "D:/文档" --no-include-content
```

**刷新方式对比：**
| 方式 | 适用场景 | 耗时 | 特点 |
|------|----------|------|------|
| 全量刷新 | 首次构建、索引损坏、大量变更 | 较长 | 完整可靠 |
| 增量刷新 | 日常维护、少量变更 | 很短 | 快速高效 |

### 2.4 搜索执行

**命令：**
```bash
python scripts/searcher.py --query "<搜索内容>" [--top-k N] [--threshold T] [--json]
```

**参数：**
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--query` | 搜索查询文本 | 必填 |
| `--top-k` | 返回结果数量 | 10 |
| `--threshold` | 相似度阈值（0-1） | 0 |
| `--json` | 输出JSON格式 | False |

**示例：**
```bash
# 基本搜索
python scripts/searcher.py --query "年结应急预案"

# 限制结果数量
python scripts/searcher.py --query "BCM调度" --top-k 20

# 设置相似度阈值
python scripts/searcher.py --query "报告" --threshold 0.7

# JSON输出
python scripts/searcher.py --query "数据分析" --json
```

### 2.5 模型管理（纯本地模式）

**命令：**
```bash
python scripts/model_manager.py [--scan] [--current] [--interactive] [--switch] [--test] [--guide]
```

**参数：**
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--scan` | 扫描本地模型 | False |
| `--current` | 显示当前使用的模型 | False |
| `--interactive` | 交互式选择模型 | False |
| `--switch <路径>` | 切换到指定模型路径 | - |
| `--test <路径>` | 测试模型加载 | - |
| `--guide` | 显示下载配置指南 | False |
| `--json` | 输出JSON格式 | False |

**示例：**
```bash
# 扫描本地模型
python scripts/model_manager.py --scan

# 显示当前模型
python scripts/model_manager.py --current

# 交互式选择模型
python scripts/model_manager.py --interactive

# 测试模型加载
python scripts/model_manager.py --test ~/.cache/huggingface/hub/models--moka-ai--m3e-base/snapshots/xxx

# 切换模型
python scripts/model_manager.py --switch ~/.cache/huggingface/hub/models--BAAI--bge-large-zh/snapshots/xxx

# 查看下载配置指南
python scripts/model_manager.py --guide
```

**扫描本地模型输出示例：**
```
本地模型列表 (共 4 个):
============================================================

[1] 💾 BAAI/bge-large-zh
    来源: huggingface_cache
    版本数: 1

[2] 💾 【当前】moka-ai/m3e-base
    来源: huggingface_cache
    版本数: 1
      - 764b537a0e50e5c7d64db883f2d2e051cbe3c64c ✓ 当前使用

[3] 📁 my-custom-model
    来源: skill_local
    版本数: 1
```

**模型存放位置：**

本技能支持两个模型存放位置：

| 位置 | 路径 | 说明 |
|------|------|------|
| HuggingFace缓存 | `~/.cache/huggingface/hub/models--<模型名>/snapshots/` | 自动检测 |
| Skill本地目录 | `<skill-dir>/models/` | 自定义模型推荐位置 |

**下载新模型指导：**

本技能为纯本地模式，不自动下载模型。请按以下步骤手动下载：

**方法1: 使用 HuggingFace CLI**
```bash
pip install huggingface_hub
huggingface-cli download moka-ai/m3e-base --local-dir ~/.cache/huggingface/hub/models--moka-ai--m3e-base/snapshots/v1
```

**方法2: 使用 Python（需要网络）**
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('moka-ai/m3e-base')
# 模型会自动缓存到 ~/.cache/huggingface/hub/
```

**方法3: 手动下载**
1. 访问 HuggingFace: https://huggingface.co/models?search=sentence-transformers
2. 找到目标模型页面，如: https://huggingface.co/moka-ai/m3e-base
3. 点击 "Files and versions" 下载所有文件
4. 放置到以下目录之一：
   - `~/.cache/huggingface/hub/models--moka-ai--m3e-base/snapshots/<版本号>/`
   - `<skill-dir>/models/moka-ai-m3e-base/`

**切换模型步骤：**

```bash
# 1. 扫描本地模型
python scripts/model_manager.py --scan

# 2. 交互式选择
python scripts/model_manager.py --interactive

# 或直接切换
python scripts/model_manager.py --switch <模型路径>
```

**切换模型后注意事项：**
- 切换模型后需要重新构建索引（全量刷新）
- 不同模型维度不同，旧索引可能不兼容
- 建议先重置向量库，再构建新索引

**常用模型推荐：**

| 模型名称 | 说明 | 维度 | HuggingFace链接 |
|----------|------|------|-----------------|
| moka-ai/m3e-base | 中文优化，推荐 | 768 | https://huggingface.co/moka-ai/m3e-base |
| BAAI/bge-large-zh | 中文高精度 | 1024 | https://huggingface.co/BAAI/bge-large-zh |
| BAAI/bge-small-zh | 中文快速 | 512 | https://huggingface.co/BAAI/bge-small-zh |
| sentence-transformers/all-MiniLM-L6-v2 | 英文 | 384 | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 |

### 2.6 向量库重置

**命令：**
```bash
python scripts/reset_vector_store.py [--status] [--reset] [--force] [--json]
```

**参数：**
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--status` | 查看向量库状态 | 默认执行 |
| `--reset` | 执行重置操作 | False |
| `--force` | 确认重置（必须） | False |
| `--json` | 输出JSON格式 | False |

**示例：**
```bash
# 查看向量库状态
python scripts/reset_vector_store.py --status

# 重置向量库（需要确认）
python scripts/reset_vector_store.py --reset --force

# JSON输出
python scripts/reset_vector_store.py --status --json
```

**状态输出示例：**
```
向量库路径: ~/.config/opencode/skills/local-semantic-search/vector_store
总大小: 4.04 MB
文件数: 2
目录数: 3
已索引文件数: 300
索引目录: D:\文档\05.模块服务\07.LTS
索引时间: 2026-04-28T15:00:00

文件列表:
  - chroma.sqlite3 (4132.0 KB)
  - metadata.json (2.4 KB)
```

**重置输出示例：**
```
向量库已重置，恢复为初始状态
删除文件: 2 个
  - chroma.sqlite3
  - metadata.json
删除目录: 3 个
  - ...

向量库已恢复为初始状态，可以重新构建索引
```

**适用场景：**
| 场景 | 说明 |
|------|------|
| 索引损坏 | 索引数据异常，搜索失败 |
| 切换模型 | 更换嵌入模型后需要重建 |
| 清理数据 | 删除旧索引，重新开始 |
| 磁盘清理 | 向量库占用空间过大 |
| 目录切换 | 索引新目录前清理旧数据 |

**注意事项：**
- 重置操作不可恢复，请谨慎使用
- 必须使用 `--force` 参数确认操作
- 重置后需要重新构建索引才能搜索

### 2.6 搜索结果格式

**文本输出：**
```
搜索完成，耗时 604ms
索引目录: D:\文档\05.模块服务\07.LTS
索引文件数: 300
找到 10 个结果:

[1] 2024年结保障-数据调度服务.docx
    路径: D:\文档\...\2024年结保障.docx
    类型: docx | 大小: 361356 bytes | 相似度: 81.05%
    修改时间: 2024-12-08T11:15:48Z
    匹配内容: 应急预案 2024年结保障...
```

**JSON输出：**
```json
{
  "results": [
    {
      "path": "D:\\文档\\...\\2024年结保障.docx",
      "name": "2024年结保障.docx",
      "type": "docx",
      "size": 361356,
      "mtime": "2024-12-08T11:15:48Z",
      "score": 0.8105,
      "matches": ["应急预案 2024年结保障..."]
    }
  ],
  "total": 10,
  "search_time_ms": 604,
  "indexed_dir": "D:\\文档\\05.模块服务\\07.LTS",
  "indexed_at": "2026-04-28T15:00:00"
}
```

---

## 第三章：本地模型切换指导

### 3.1 支持的嵌入模型

本技能支持多种中文/英文嵌入模型：

| 模型 | 说明 | 特点 | 适用场景 |
|------|------|------|----------|
| moka-ai/m3e-base | 中文优化模型 | 中英文双语，768维 | 中文文档搜索（默认） |
| BAAI/bge-large-zh | 中文大模型 | 中文优化，1024维 | 高精度中文搜索 |
| paraphrase-multilingual-MiniLM-L12-v2 | 多语言模型 | 支持50+语言 | 多语言文档 |

### 3.2 查看本地缓存模型

```bash
ls ~/.cache/huggingface/hub/
```

或 Windows：
```cmd
dir %USERPROFILE%\.cache\huggingface\hub
```

### 3.3 切换模型方法

**方法1：修改脚本配置**

编辑 `scripts/indexer.py` 和 `scripts/searcher.py`，修改模型配置：

```python
# 找到以下配置行
LOCAL_MODEL_PATH = Path.home() / '.cache' / 'huggingface' / 'hub' / 'models--moka-ai--m3e-base' / 'snapshots' / '<版本号>'
MODEL_NAME = 'moka-ai/m3e-base'

# 切换到 bge-large-zh
LOCAL_MODEL_PATH = Path.home() / '.cache' / 'huggingface' / 'hub' / 'models--BAAI--bge-large-zh' / 'snapshots' / '<版本号>'
MODEL_NAME = 'BAAI/bge-large-zh'
```

**方法2：下载新模型**

```bash
# 使用 Python 下载模型
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-zh')"
```

下载后按方法1修改配置。

**方法3：命令行参数（未来支持）**

计划支持通过命令行参数指定模型：
```bash
python scripts/indexer.py --dir <目录> --model BAAI/bge-large-zh
```

### 3.4 模型切换注意事项

1. **重建索引**：切换模型后需要全量刷新索引，因为向量维度可能不同
2. **兼容性**：确保新模型与 ChromaDB 兼容
3. **性能**：大模型（如 bge-large-zh）精度更高但速度稍慢
4. **磁盘空间**：每个模型约500MB-1GB

### 3.5 模型切换完整流程

```bash
# 1. 下载新模型
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-zh')"

# 2. 查看模型版本号
ls ~/.cache/huggingface/hub/models--BAAI--bge-large-zh/snapshots/

# 3. 修改脚本配置
# 编辑 indexer.py 和 searcher.py

# 4. 全量刷新索引
python scripts/indexer.py --dir <目录> --force

# 5. 测试搜索
python scripts/searcher.py --query "测试内容"
```

---

## 第四章：支持的文件类型

| 类型 | 扩展名 | 内容提取方式 |
|------|--------|--------------|
| 文本文件 | .txt | 直接读取 |
| Markdown | .md | 直接读取 |
| PDF | .pdf | PyPDF2提取文本 |
| Word | .docx | python-docx提取段落 |
| PowerPoint | .pptx | python-pptx提取幻灯片 |

---

## 第五章：常见问题

### Q1: 如何在CLI中触发搜索？
直接说"帮我在本地搜索..."或"本地搜索..."即可触发。

### Q2: 如何知道目录是否已索引？
CLI中说"检查索引状态"，或命令行运行 `check_index.py`。

### Q3: 全量刷新和增量刷新的区别？
- 全量：删除旧索引，重建所有文件，耗时长但完整
- 增量：只更新变化的文件，速度快但需要已有索引

### Q4: 索引过期了怎么办？
CLI中说"刷新索引"，选择增量或全量刷新。

### Q5: 如何切换嵌入模型？
修改脚本配置中的 MODEL_NAME 和 LOCAL_MODEL_PATH，然后全量刷新索引。

### Q6: 切换索引目录后旧索引会丢失吗？
是的，每次索引新目录会覆盖旧索引。

### Q7: 搜索速度如何？
首次搜索约500-700ms（含模型加载），后续搜索更快。

### Q8: 向量库占用多少空间？
约每1000文件 50-100MB。

### Q9: 如何重置向量库？
CLI中说"重置向量库"，或命令行运行 `reset_vector_store.py --reset --force`。

### Q10: 重置向量库后还能搜索吗？
不能，重置后需要重新构建索引才能搜索。

---

## 第六章：文件结构

```
local-semantic-search/
├── SKILL.md              # 技能说明文档
├── requirements.txt      # Python依赖
├── scripts/
│   ├── check_index.py   # 索引状态检查脚本
│   ├── indexer.py       # 索引构建脚本（支持全量/增量）
│   ├── searcher.py      # 搜索执行脚本
│   └── reset_vector_store.py  # 向量库重置脚本
├── references/
│   └── usage.md         # 使用指南（本文件）
├── vector_store/        # 向量数据库存储
│   ├── chroma.sqlite3   # ChromaDB数据库
│   └── metadata.json    # 索引元数据
└── evals/
    └── evals.json       # 测试用例定义
```

---

## 更新日志

### v1.5.0 (2026-04-28)
- ✅ 调整为纯本地模式，移除所有联网功能
- ✅ 移除自动下载模型功能
- ✅ 模型扫描仅扫描本地缓存和skill本地目录
- ✅ 新增模型下载配置指南（--guide）
- ✅ 新增skill本地模型目录支持
- ✅ 模型加载失败提示纯本地解决方案
- ✅ 更新模型管理章节为纯本地模式

### v1.4.0 (2026-04-27)
- ✅ 新增模型管理功能
- ✅ 新增 model_manager.py 脚本
- ✅ 支持扫描本地缓存模型
- ✅ 支持交互式选择模型
- ✅ 支持下载新模型
- ✅ 模型加载失败自动提示解决方案
- ✅ 新增CLI模型管理触发词
- ✅ 新增命令行模型管理章节
- ✅ 新增下载新模型指导

### v1.3.0 (2026-04-27)
- ✅ 新增向量库重置功能
- ✅ 新增 reset_vector_store.py 脚本
- ✅ 新增重置向量库CLI触发词和对话示例
- ✅ 新增常见问题Q9、Q10
- ✅ 更新文件结构说明

### v1.2.0 (2026-04-26)
- ✅ 新增CLI自然语言操作章节
- ✅ 新增命令行操作章节
- ✅ 新增本地模型切换指导
- ✅ 补充完整对话示例
- ✅ 更新常见问题解答

### v1.1.0 (2026-04-24)
- ✅ 新增目录索引状态检查功能
- ✅ 新增全量/增量刷新选择
- ✅ 新增 check_index.py 脚本
- ✅ 更新 SKILL.md 工作流程说明

### v1.0.0 (2026-04-23)
- ✅ 实现语义搜索功能
- ✅ 支持 txt, md, pdf, docx, pptx 格式
- ✅ 使用 ChromaDB 向量数据库
- ✅ 使用 m3e-base 中文嵌入模型
- ✅ 支持余弦距离度量
- ✅ JSON 和文本两种输出格式
