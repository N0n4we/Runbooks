# 反模式：通过临时文件传递本可按值传递的 JSON

## 核心原则

**脚本需要的是数据，就优先让接口接收数据，而不是要求调用方先把数据写成文件。**

当 JSON 已由上游生成、规模较小，且没有持久化需求时，把下游输入设计成文件路径，会引入不必要的文件系统依赖。问题的重点不是多一次磁盘 I/O，而是调用方必须额外维护接口之外的状态。

路径虽然是显式参数，但 JSON 内容并不包含在参数值中，而是存在于外部文件系统。文件是否存在、是否可读、内容是否已写完、读取期间是否变化，以及谁负责清理，都成为调用的隐式前置条件。

这里的反模式是**不必要地把临时文件变成数据交换契约**，不是说所有文件输入接口都是错误的。

## 案例：Cloud 镜像更新脚本

来源：`platform-manager-stack` 仓库中的：

- `.github/workflows/cloud.yaml`
- `.github/scripts/parse-cloud-images.sh`
- `.github/scripts/update-cloud-images.sh`

记录时的调用方式：

```bash
set -euo pipefail

parsed_images="$RUNNER_TEMP/cloud-images.json"
.github/scripts/parse-cloud-images.sh "$IMAGES_INPUT" > "$parsed_images"

update_result="$RUNNER_TEMP/cloud-update-result.json"
.github/scripts/update-cloud-images.sh \
  "$DEPLOY_ENVIRONMENT" "$parsed_images" > "$update_result"
```

其中 updater 的第二个参数是 JSON 文件路径，而不是 JSON 文本：

```bash
environment="$1"
parsed_images="$2"

[[ -f "$parsed_images" ]] || fail 'parsed images JSON file does not exist.'
target_family="$(jq -er '.target_family' "$parsed_images")"
```

这使本来简单的关系：

```text
镜像字符串 → parser → JSON 数据 → updater → 结果 JSON
```

变成：

```text
parser → 调用方创建并写入文件 → updater 打开并读取文件
```

但这个场景中，两个脚本在同一个 workflow step 内顺序执行，JSON 规模受 allowlist 和目标去重规则限制，也没有必须落盘的业务要求。

## 为什么这是问题

1. **数据依赖变成外部状态依赖。** 相同路径参数不代表相同输入内容；文件被修改、删除或权限改变，都可能影响调用。
2. **调用方承担额外职责。** 除了提供 JSON，还要选择临时目录、命名文件、完成写入、管理生命周期和清理。
3. **测试增加无关准备工作。** 仅测试 JSON 输入校验，也要创建文件 fixture，不能直接传入一个字符串完成调用。
4. **复用与组合更困难。** 其他程序已经持有 JSON 时，仍被迫先落盘；路径也不能直接跨 runner 或隔离环境使用。
5. **多次读取依赖文件保持不变。** 当前 updater 多次调用 `jq` 读取同一路径，默认这些读取看到的是同一份数据。临时文件若没有隔离或被其他进程修改，这个假设就可能失效。

“文件方便调试”“会多次读取 JSON”“当前测试已经使用文件”都不足以单独证明文件接口是必要的。需要调试时，调用方可以选择保存副本，而不是让落盘成为每次调用的前提。

## 推荐设计

### 小型 JSON：显式按值传入

将 updater 的接口定义为：

```text
update-cloud-images.sh <environment> <parsed-images-json>
```

第二个参数是 **JSON 文本**，不是 shell 变量名，也不是通过 `export` 隐式提供的环境变量。

目标调用方式：

```bash
set -euo pipefail

parsed_images_json="$(
  .github/scripts/parse-cloud-images.sh "$IMAGES_INPUT"
)"

update_result_json="$(
  .github/scripts/update-cloud-images.sh \
    "$DEPLOY_ENVIRONMENT" "$parsed_images_json"
)"

target_family="$(jq -er '.target_family' <<< "$update_result_json")"
kustomization_file_path="$(
  jq -er '.kustomization_file_path' <<< "$update_result_json"
)"
printf 'restart_targets=%s\n' \
  "$(jq -c '.restart_targets' <<< "$update_result_json")" >> "$GITHUB_OUTPUT"
```

以上是建议的接口，**原脚本只接受文件路径，不能不改实现就使用此调用方式**。

updater 内部相应调整：

- 用 `parsed_images_json="$2"` 明确保存 JSON 文本。
- 删除输入文件存在性检查。
- 将 `jq ... "$parsed_images"` 改成 `jq ... <<< "$parsed_images_json"`。
- 保留参数数量、环境、JSON 结构、重复目标及 overlay 等校验；改变传输方式不等于取消校验。
- 继续先校验全部目标，再修改资源；stdout 只输出结果 JSON，错误写 stderr，失败返回非零退出码。
- 同步修改 workflow 和测试调用，不要自动猜测第二个参数是文件路径还是 JSON。

调用时始终使用双引号传递 JSON，并通过 `jq` 解析；不要使用 `eval` 或将数据拼接成可执行表达式。

### 更通用的 CLI：可以设计为 stdin 输入

对于较大 JSON，或不适合出现在命令行参数中的敏感数据，可明确设计为从 stdin 接收。需要重复解析时，消费者应先读取并保存一份输入，再交给多个 `jq` 调用，不能假设同一 stdin 可以反复读取。

无论使用位置参数还是 stdin，核心都是：**让数据直接进入消费者，不要求调用方先创建中间文件。**

## 边界与例外

文件接口在以下场景中仍然合理：

- 输入本身就是用户选定的配置文件或磁盘上的持久化文档。
- 数据很大，需要流式处理、随机访问，或避免受命令行参数长度限制。
- 文件是明确要求保存、审计、归档或跨 job 传递的 artifact。
- 外部工具只支持文件输入；此时可将临时文件适配封装在边界层。

采用文件接口时，应明确记录格式、路径解析规则、权限、写入完成条件、读取期间的稳定性和清理责任。跨 job 还需要显式上传、下载 artifact，不能只传本地路径。

JSON 改为按值传入，也不意味着整个 updater 变成纯函数：它仍需要读取、更新仓库里的 Kubernetes 资源，这些是其业务职责。资源更新的集成测试仍应使用隔离目录。本次消除的是**中间 JSON 的非必要文件依赖**。

## Review 检查清单

- [ ] 接口真正需要的是文件，还是文件里的数据？
- [ ] 是否仅为两个进程交换小型 JSON，就要求调用方创建临时文件？
- [ ] 文件依赖是否有明确业务必要，而非只为方便调试或沿用现有实现？
- [ ] 能否通过显式参数或 stdin 传入数据，并让 stdout 返回结果？
- [ ] 是否保留了输入校验、正确引用和明确的错误处理？
- [ ] 是否清楚区分了可消除的中间文件依赖与必要的业务资源读写？
