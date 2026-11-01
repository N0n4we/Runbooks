# 反模式：为局部需求在共享层复活长期失效的兼容逻辑

## 核心原则

**当一个分支、角色或兼容路径已经长期没有被实际命中时，任何让它重新生效的改动都应被视为一次行为恢复或迁移，而不是局部功能实现中的无害兼容。**

在恢复之前，必须先回答：

- 该行为最初解决什么问题？
- 它是被有意移除，还是在重构中意外丢失？
- 长期未生效期间，调用方、部署矩阵和运行环境是否已经改变？
- 重新生效会命中哪些共享调用方和原本休眠的代码？
- 被重新激活的角色是否与当前角色互斥，是否会造成重复副作用？

本案例包含三个相互叠加的反模式：

- **Blind compatibility resurrection**：没有先追溯历史契约，就重新激活长期失效的兼容行为。
- **Scope inversion**：为一个局部功能解决问题，却把补丁放进所有工具共用的基础设施层。
- **Role aliasing**：通过 alias 让同一资源同时拥有原本互斥的角色，改变未参与本次需求的执行路径。

这里不是说旧行为永远不能恢复，也不是说历史越久就越不应该修。需要审计的是：**是否把一次有显著行为面的恢复，误当成一个无需调查历史和调用方的局部补丁。**

## 案例：重新激活 `agent1` 兼容导致共享工作流回归

来源：`cloud-cap/semaphore-script`，涉及：

- `dm_tools/dm_tools.py`
- `semaphore_utils.py`
- `ansible_scripts/download_logs_v4.yaml`
- `ansible_scripts/download_trace_logs.yaml`
- `ansible_scripts/release_mem_v4.yaml`

### 最初实现：按部署类型互斥选择角色

早期提交 `a8f0315999f2` 和 `ba881a370ac5` 中，inventory 生成器根据 `deploymentType` 选择 Agent 的 inventory hostname。

简化后的结构如下：

```yaml
# basic
agent:
  hosts:
    agent1:
      ansible_host: <agent-public-ip>

# dedicated
agent:
  hosts:
    agent0:
      ansible_host: <agent-public-ip>
emqx:
  hosts:
    emqx0:
      ansible_host: <emqx-private-ip>
```

对应 playbook 使用：

```yaml
- name: dedicated agent
  hosts: agent0

- name: basic agent
  hosts: agent1

- name: dedicated emqx
  hosts: emqx
```

这套设计表达的是互斥路由：

- basic Agent 只命中 `agent1`。
- dedicated Agent 只命中 `agent0`。
- dedicated 的 EMQX 节点命中 `emqx`。
- 同一台 Agent 不会同时执行 basic 和 dedicated 的业务任务。

`agent0` / `agent1` 不是理想的语义化角色名，但旧实现至少保持了角色互斥。

### 统一重构：兼容逻辑丢失

提交 `4c3314bedf7b`（`feat(semaphore): refactor the code to run dm_tool on semaphore`）引入统一的 DM 工具和 inventory 构建器。新的资源创建路径固定传入：

```python
ensure_ssh_key_and_inventory(
    semaphore,
    deployment_id,
    deployment_info,
    group_name='agent0',
)
```

inventory 构建器随后无条件生成：

```ini
[agent0]
<deployment-id> ansible_host=<agent-public-ip>
```

但是重构后的 V4 playbook 仍保留 `hosts: agent1`。于是从该提交开始：

- V4 basic 日志下载的 `agent1` 采集 play 不再命中任何主机。
- basic trace 下载的 `agent1` play 不再命中任何主机。
- V4 basic 内存释放错误地命中 `agent0` 的 dedicated 分支，并跳过 `agent1` 的 basic 分支。

这不是运行时偶发问题，而是 inventory 模型与 playbook 主机选择器之间的静态错位。

### 近一年后：局部功能重新激活共享兼容行为

约十个月后，提交 `4712792a214a` 为日志下载增加时间范围和 systemd 日志支持。为了让 V4 basic 的 `hosts: agent1` 再次命中主机，改动在共享 inventory 构建器中加入：

```ini
[agent1:children]
agent0
```

这个 alias 的含义不是“将 basic Agent 正确分类为 agent1”，而是：

```text
同一台 Agent 同时属于 agent0 和 agent1
```

因为 `semaphore_utils.py` 的 inventory 构建器被所有 DM 工具共用，这个为日志功能加入的兼容行为同时影响了 `mem_release`。

`release_mem_v4.yaml` 的两个 play 都包含实际副作用：

```yaml
- name: dedicated agent
  hosts: agent0
  tasks:
    - include_tasks: tasks/release_system_cache.yaml

- name: basic agent
  hosts: agent1
  tasks:
    - include_tasks: tasks/release_system_cache.yaml
```

alias 使 basic 主机依次执行两个 play，造成：

- 两次释放系统缓存。
- 两次清理 apt 数据。
- 两次轮转和压缩 journald。
- 两次重启 systemd-journald。
- 两次重启 agent，并额外重启 basic 需要的服务。

因此，这次实现虽然修复了目标日志路径，却重新激活了一个在共享层失效已久的角色关系，并让无关工具出现新的生产副作用。

## 为什么这是反模式

### 1. 长期失效是需要调查的证据，不是可直接恢复的规格

一个分支长期未命中，可能意味着：

- 重构时意外丢失了兼容逻辑。
- 该部署组合已经不再受支持。
- 调用方已经迁移到另一条路径。
- 原 playbook 成为未删除的死代码。
- 生产上一直存在潜在缺陷，只是缺少观测。

仅凭代码中仍存在 `hosts: agent1`，不能确定正确修复就是重新创建 `agent1`。必须结合历史实现、当前部署矩阵、生产调用和期望副作用确认。

**时间跨度本身不证明旧行为应该删除，也不证明旧行为应该恢复；它要求更高等级的证据。**

### 2. 局部需求不应默默改变共享基础设施语义

目标需求是日志时间范围和 systemd 日志下载，但修复落在所有 DM 工具共用的 inventory 构建器中。

需要警惕这种链条：

```text
某个 playbook 没命中目标主机
    → 修改共享 inventory
    → 所有 playbook 获得新的组关系
    → 未审计的旧分支开始执行
```

共享函数中的少量改动，可能比业务 playbook 中的大段修改拥有更大的行为面。评审不能按修改行数判断风险。

### 3. alias 可能破坏原本互斥的角色

`[agent1:children] agent0` 在语法上合法，但它把“basic”和“dedicated Agent”从分类关系变成叠加关系。

如果两个角色只提供相同的只读变量，叠加可能无害；如果两个角色各自执行重启、删除、发布、扩缩容或权限修改，叠加通常意味着重复或冲突执行。

审计 alias 时必须问：

- 这是能力叠加，还是部署类型分类？
- 两个组是否在同一个 playbook 中分别承担业务任务？
- 同一物理主机匹配两个组后，每个 task 会执行几次？
- 任务是否幂等？即使模块幂等，重启、清缓存等运行时副作用是否可重复？

不能只验证 alias 让目标 play 从 0 台主机变成 1 台主机，还要检查它是否让其他 play 从 1 次变成 2 次。

### 4. 长期未执行的代码不能按已验证代码对待

一个 play 在近一年内没有命中主机，即使代码一直存在，也不能据此认为它经过持续生产验证。

重新激活时应把它当成新代码审查：

- 模块和变量是否仍兼容当前 Ansible 版本？
- 路径、服务名和权限是否仍有效？
- 与当前共享 task 的组合是否仍正确？
- 失败、清理和重试语义是否符合当前编排？
- 是否会与已经接管该职责的新路径重复？

“这段代码以前就在”不能替代重新验证。

### 5. 兼容策略分散后，很难证明有效范围

一种常见补丁是：调用方根据工具和版本传入 `include_agent1_alias`，构建器再根据 `deploymentType` 决定是否真正输出 alias。

这会把策略拆散为：

```text
工具判断：调用方
版本判断：调用方
部署类型判断：共享构建器
主机匹配语义：playbook
```

任何一处单独阅读都看不到完整生效范围。新增工具、版本或部署类型时，也容易遗漏同步修改。

若必须保留兼容开关，应把完整策略集中为可命名、可测试的决策，例如：

```python
def inventory_roles_for(tool_type, major_version, deployment_type):
    ...
```

返回语义化角色集合，而不是让多个层分别拼出一个布尔结果。

### 6. 修复症状会掩盖领域模型错误

真正的问题是 inventory 使用 `agent0` / `agent1` 表达部署类型，而统一构建器又把它们当成固定组名。alias 只让不一致暂时可运行，没有澄清角色模型。

更合理的长期模型通常是：

- 使用 `basic_agent`、`dedicated_agent`、`emqx` 等语义化组名。
- 或由一个集中决策函数根据工具、版本和部署类型生成互斥角色。
- playbook 明确声明需要哪种角色，不依赖编号暗示业务含义。

## 正确处理方向

### 第一步：先做历史考古

在修改兼容层前，至少检查：

```bash
jj log -- <相关文件>
jj show <可疑重构提交> --git
jj file show -r '<提交>-' <历史文件路径>
rg -n 'agent0|agent1|hosts:|group_name|deploymentType' <相关目录>
```

需要找出：

- 不一致从哪个提交开始。
- 修改前的显式行为是什么。
- playbook 和 inventory 是否在同一提交中一起迁移。
- 后续是否有提交或文档说明废弃某个部署组合。

不要只比较当前提交与直接父提交。直接父提交可能已经携带多月甚至多年的潜在回归。

### 第二步：建立完整行为矩阵

至少列出：

| 工具 | EMQX 版本 | 部署类型 | inventory 角色 | 命中的 plays | 关键副作用 |
| --- | --- | --- | --- | --- | --- |
| `dl_logs` | V4 | basic | 待确认 | agent/basic/emqx | SSH 转发、采集、清理 |
| `dl_logs` | V4 | dedicated | 待确认 | agent/basic/emqx | SSH 转发、多节点采集 |
| `dl_logs` | V5/V6 | basic/dedicated | 待确认 | agent/emqx | Gateway 与 EMQX 日志 |
| `dl_trace_logs` | 各版本 | basic/dedicated | 待确认 | agent/basic/emqx | trace 采集 |
| `mem_release` | V4 | basic/dedicated | 待确认 | agent/basic/emqx | 清缓存、重启服务 |

矩阵中的“待确认”必须通过当前产品支持范围和实际 playbook 填实，不能仅从名称推断。

### 第三步：明确是在恢复、迁移还是删除

根据证据选择一种策略：

1. **恢复旧契约**：旧行为仍是当前受支持规格。应恢复互斥路由，并补齐所有受影响调用方的回归测试。
2. **迁移到新模型**：旧角色名已经不适合当前架构。应同步修改 inventory 和 playbook，使用明确角色完成一次可审计迁移。
3. **删除失效路径**：对应部署组合已不受支持。应删除死分支或显式拒绝请求，而不是保留看似支持但永远不执行的 play。
4. **临时兼容 shim**：只有无法立即迁移时使用。必须严格限定工具、版本、部署类型，记录移除条件，并测试所有共享调用方不受影响。

### 第四步：保持角色互斥

本案例中，若业务规格仍与旧实现一致，目标关系应接近：

```text
V4 basic:
    basic Agent → basic 角色一次

V4 dedicated / dedicatedFlex:
    Agent → dedicated Agent 角色一次
    EMQX nodes → emqx 角色一次
```

不要通过 alias 让 basic Agent 同时拥有 basic 和 dedicated 角色。若某些通用任务两者都需要，应抽取通用 task，由两个互斥角色各自显式引用，而不是让一台主机匹配两个完整业务 play。

### 第五步：验证主机匹配和副作用次数

静态语法检查不足以发现这类问题。至少运行：

```bash
ansible-playbook -i <basic-inventory> --list-hosts <playbook>
ansible-playbook -i <dedicated-inventory> --list-hosts <playbook>
ansible-playbook -i <inventory> --list-tasks <playbook>
```

对于有副作用的 task，还要在隔离环境记录执行次数。验收标准包括：

- 每台物理主机只命中预期业务角色。
- 新目标路径从 0 次变为 1 次，而不是从 0 次变为 2 次。
- 无关工具的主机匹配集合不变。
- 重启、删除、缓存释放和配置修改不会重复执行。
- 所有支持的版本与部署类型都有明确结果。
- 不支持的组合显式失败或跳过，并有文档依据。

## 不应误判的合理场景

以下情况不应仅因“恢复旧逻辑”或“使用 alias”就直接判为缺陷：

- 历史行为被确认是受支持契约，重构中意外丢失，现在有完整部署矩阵和测试证明恢复正确。
- 安全或数据完整性缺陷长期存在，修复必须重新启用某个校验或保护路径。
- alias 表达的是可叠加能力，而不是互斥部署类型；所有匹配任务均验证为可组合且执行次数正确。
- 临时兼容层有明确调用边界、负责人、删除条件和回归测试。
- 共享层本身就是该策略的正式所有者，并且接口表达的是完整领域决策，而不是某个调用方的特例。

关键区别不在于代码是不是旧的，而在于恢复行为是否经过明确授权、完整建模和跨调用方验证。

## Review 检查清单

- [ ] 此修改是否让一个长期为 0 hosts、未调用或条件恒假的分支重新生效？
- [ ] 是否查看了不一致首次出现的提交，而不只是当前 diff 的直接父版本？
- [ ] 旧行为是当前规格、历史实现、死代码，还是未知状态？证据是什么？
- [ ] 需求是局部功能，但改动是否落在共享 inventory、路由、鉴权、注册表或基础 helper 中？
- [ ] 是否枚举了该共享函数的全部调用方？
- [ ] 是否建立了工具 × 版本 × 部署类型的行为矩阵？
- [ ] 一个 alias、继承或 fallback 是否让同一资源同时拥有原本互斥的角色？
- [ ] 是否比较修改前后每个 play / handler / callback 的主机集合？
- [ ] 是否检查了重启、删除、发布、扩缩容、清缓存等非只读副作用的执行次数？
- [ ] 长期未运行的分支是否按新代码重新验证，而不是因“以前存在”就默认可信？
- [ ] 兼容判断是否分散在调用方、共享 helper 和 playbook 多层？
- [ ] 新增的布尔开关是否在掩盖应由领域模型表达的角色选择？
- [ ] 能否恢复互斥路由或使用语义化角色，而不是增加双重身份？
- [ ] 无关工具是否有回归测试，证明主机匹配和副作用没有变化？
- [ ] 若是临时 shim，是否记录了精确范围、移除条件和负责人？
- [ ] 不支持的历史组合是否显式拒绝，而不是继续保留永远不命中的代码？

## 搜索入口与审计方法

可以先搜索以下信号：

```bash
rg -n 'alias|compat|legacy|children|include_.*alias|group_name' <待审计目录>
rg -n 'hosts:|groups\[|hostvars\[' <playbook目录>
rg -n 'deploymentType|major_version|version' <inventory和调度代码>
rg -n 'restart|reboot|delete|state: absent|drop_caches|vacuum|mask' <相关tasks>
```

然后检查历史：

```bash
jj log -- <共享helper> <受影响playbook>
jj show <引入不一致的提交> --git
jj file show -r '<引入不一致的提交>-' <旧实现路径>
```

最后对每种 inventory 执行主机匹配验证。不能只运行 `--syntax-check`；语法正确并不代表主机集合正确。

建议在评审记录中保留一张差异表：

| 调用场景 | 修改前命中主机 | 修改后命中主机 | 新增执行任务 | 是否符合规格 |
| --- | --- | --- | --- | --- |
| 目标功能 |  |  |  |  |
| 同版本其他部署类型 |  |  |  |  |
| 其他版本 |  |  |  |  |
| 共享 helper 的其他调用方 |  |  |  |  |

任何“修改后命中主机增加”的行都需要说明原因，尤其是有副作用的 play。

## 审计结论如何表述

不要只写“这是老代码，所以不应该恢复”，也不要只写“alias 影响范围太大”。应指出历史断点、共享传播路径和具体副作用，例如：

> 此变更为修复 V4 basic 日志下载，在共享 inventory 构建器中将 `agent0` alias 到 `agent1`。历史上 basic 与 dedicated 角色是互斥的；统一重构后该映射已失效近一年。当前 alias 会使共享 helper 的其他调用方也重新命中 `agent1`，其中 `release_mem_v4` 的 `agent0` 和 `agent1` plays 都包含清缓存与服务重启，因此同一 basic 主机会执行两套副作用。请先确认当前支持矩阵，并恢复互斥角色路由或把兼容限定在经过验证的调用边界；不能仅用目标日志路径成功来证明该共享修改安全。

如果历史调查确认该兼容应恢复，也应将结论写成一次明确的行为恢复：列出支持范围、受影响调用方、迁移策略和测试，而不是把它描述成内部实现细节。

## 最小记忆规则

以后看到以下组合时，应立即触发历史与调用方审计：

```text
长期未命中的旧分支
+ 为新需求修改共享 helper
+ alias / fallback / compatibility flag 让旧分支重新生效
```

此时默认问题不是“怎样让目标分支跑起来”，而是：

```text
为什么它长期没有运行？
现在恢复的是当前契约，还是过期实现？
共享层还有谁会因此开始运行？
同一资源会不会同时承担互斥角色？
```

只有这些问题得到证据支持后，兼容代码才可以合入。
