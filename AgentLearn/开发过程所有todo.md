# 开发过程所有TODO

## TODO 列表

### 2.Memory/agent_memory.py:13
**TODO 内容**: 这个记忆实现方式很简单，直接写到一个md文件里，更优的做法为放到向量数据库中

**处理结果**: 

---

### 2.Memory/agent_memory.py:97
**TODO 内容**: 还需要加上异常处理

**处理结果**: 

---

### 2.Memory/agent_memory.py:129
**TODO 内容**: 这里取后50行有待修改

**处理结果**: 

---

### 2.Memory/agent_memory.py:230
**TODO 内容**: 这个参数要用户给，这里比较挫，后面会优化

**处理结果**: 

---

### 2.Memory/agent_memory.py:266
**TODO 内容**: 4.14 凌晨2点遗留问题：1.模型响应要设置超时时间控制，2.写memory.md文件后pycharm打开是乱码，3.模型本身能力不足，生成的工具调用或者命令行都不是很对

**处理结果**: 

---

### 3.skills-and-mcp/agent_skill_mcp.py:86
**TODO 内容**: 这里客户端后续要剥离出来，不在这里初始化，在一个编排类里面初始化

**处理结果**: 

---

### 3.skills-and-mcp/agent_skill_mcp.py:403
**TODO 内容**: 后续修改为在agent loop前后进行打开和关闭，不要让外界感知

**处理结果**: 

---

### 3.skills-and-mcp/agent_skill_mcp.py:412
**TODO 内容**: 新建一个编排类，由这个编排类来控制Agent的运行

**处理结果**: 

---

### 3.skills-and-mcp/agent_skill_mcp.py:413
**TODO 内容**: 任务完成得不好，考虑设计一个评价器，调整温度重新生成

**处理结果**: 

---

### 3.skills-and-mcp/agent_skill_mcp.py:414
**TODO 内容**: 实现后台定时任务，agent自主行动，类似车机上车后自动打开空调等

**处理结果**: 

---

### 3.skills-and-mcp/agent_skill_mcp.py:415
**TODO 内容**: 记忆系统修改，维护两个记忆md文档，一个放未压缩的，一个放压缩的，运行时写两个文件，load记忆时优先load压缩的，再结合RAG做运行时检索旧记忆

**处理结果**: 

---

### 3.skills-and-mcp/agent_skill_mcp.py:419-433
**TODO 内容**: 子agent，两种实现方式：
1、隐式定义：一个类似读写文件、执行bash命令等的工具，在工具中另起一个Agent对象，通过提示词区分角色，该agent临时启用，生命周期为本次对话，任务完成就丢失
	需要扩展Agent对象的属性，新增is_main_agent、agent_name、agent id等属性，

2、显式定义：针对特定任务预先定义一个独立的子agent，独立的设定，可用工具列表和主agent存在部分交集，可能完全继承所有工具，也可能有主agent没有的工具，
	记忆可以存到磁盘里做永久，下次类似的任务唤起该子agent时能load上一次的工作记忆

二者区别：
对比项	 |    隐式    |      显式
创建方式  | agent运行时动态创建 | 预先根据特定任务定义特定agent，代码/配置文件定义
生命周期  | 临时创建，任务完成即销毁 | 可复用，类似于程序可永久存活在磁盘
有无状态  |  有状态   |	无状态
可控性	| 灵活，可控性弱 | 可控性强，输出可预测
适用场景	| 不确定性任务，由Agent自主决策 | 流程可固化的企业工作流，可配合Skill一起实现

**处理结果**: 

---

### 4.subAgent-and-team/agent_subAgent_team.py:487
**TODO 内容**: 新建一个编排类，由这个编排类来控制Agent的运行，需要剥离Agent的mcp_server属性

**处理结果**: 

---

### 4.subAgent-and-team/agent_subAgent_team.py:488
**TODO 内容**: 任务完成得不好，考虑设计一个评价器，调整温度重新生成

**处理结果**: 

---

### 4.subAgent-and-team/agent_subAgent_team.py:489
**TODO 内容**: 实现后台定时任务，agent自主行动，类似车机上车后自动打开空调等

**处理结果**: 

---

### 4.subAgent-and-team/agent_subAgent_team.py:490
**TODO 内容**: 记忆系统修改，维护两个记忆md文档，一个放未压缩的，一个放压缩的，运行时写两个文件，load记忆时优先load压缩的，再结合RAG做运行时检索旧记忆

**处理结果**: 

---

### 4.subAgent-and-team/agent_subAgent_team.py:491
**TODO 内容**: sub agent的记忆怎么处理

**处理结果**: 

---

### 4.subAgent-and-team/agent_subAgent_team.py:495-509
**TODO 内容**: 子agent，两种实现方式：
1、隐式定义：一个类似读写文件、执行bash命令等的工具，在工具中另起一个Agent对象，通过提示词区分角色，该agent临时启用，生命周期为本次对话，任务完成就丢失
	需要扩展Agent对象的属性，新增is_main_agent、agent_name、agent id等属性，

2、显式定义：针对特定任务预先定义一个独立的子agent，独立的设定，可用工具列表和主agent存在部分交集，可能完全继承所有工具，也可能有主agent没有的工具，
	记忆可以存到磁盘里做永久，下次类似的任务唤起该子agent时能load上一次的工作记忆

二者区别：
对比项	 |    隐式    |      显式
创建方式  | agent运行时动态创建 | 预先根据特定任务定义特定agent，代码/配置文件定义
生命周期  | 临时创建，任务完成即销毁 | 可复用，类似于程序可永久存活在磁盘
有无状态  |  有状态   |	无状态
可控性	| 灵活，可控性弱 | 可控性强，输出可预测
适用场景	| 不确定性任务，由Agent自主决策 | 流程可固化的企业工作流，可配合Skill一起实现

**处理结果**: 

---

## 统计信息
- 总 TODO 数量: 18 个
- 涉及文件: 3 个 (agent_memory.py, agent_skill_mcp.py, agent_subAgent_team.py)
- 最后更新: 2026/04/28