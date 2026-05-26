# LoraCI

Anima LoRA training - push toml to GitHub, training runs on your own GPU host.

> Training code migrated from `github.com/gazingstars123/Anima-Standalone-Trainer`

## Workflow

```
push to main (GitHub)
 │
 ▼
GitHub Actions (.github/workflows/train.yml)
 ├── SSH 到训练服务器
 ├── git fetch + reset --hard origin/main
 └── bash scripts/run_training.sh   (秒级返回)
     │
     └── 如果 queue_runner 已在跑 → 直接退出
         否则 setsid nohup queue_runner.sh & 退出
            │
            ▼
        queue_runner.sh (后台长跑，独立 session)
         循环:
          ├── git fetch + reset --hard
          ├── bash scripts/download_models.sh   (缺失基模才下载)
          ├── 扫 tasks/*.toml → 找下一个无 marker 的
          ├── 同名目录 tasks/<name>/ 不存在或空 → 发"missing dataset"邮件 → 跳过
          ├── 否则 python train.py --config_file tasks/<name>.toml
          │   └── train.py 自己发"complete" / "failed"邮件
          └── 写 marker → 继续下一个 / 队列空则退出
```

## 仓库结构

```
tasks/                        # 任务队列（用户提交 toml 到这里）
  001_topic.toml          # ✅ git 跟踪
  001_topic/              # ❌ git 不跟踪（训练集）
    img1.png
    img1.txt
    ...
  002_experiment_a.toml
  002_experiment_a/
    ...
models/base/                  # ❌ 基模（download_models.sh 自动下载）
logs/                         # ❌ 训练日志、preflight 日志、queue_runner.log
  done/
    001_topic__<sha8>.done             # 跑过的标记
    002_experiment_a__<sha8>.missing_dataset
  queue_runner.pid            # 调度器 PID 锁
  preflight_*.log
  train_<basename>_<ts>.log
  queue_runner.log
scripts/
  run_training.sh             # CI 入口（启动 queue_runner）
  queue_runner.sh             # 调度器：扫 tasks 队列、按序跑
  download_models.sh          # 缺失基模下载（需先填入真实 URL）
  notify_email.py             # SMTP 发件器（shell + python 共用）
train.py                      # 单任务入口（被 queue_runner 调用）
anima_train_network.py        # AnimaNetworkTrainer
train_network.py              # NetworkTrainer base
library/                      # 训练库
networks/                     # LoRA 实现
configs/                      # tokenizer 配置
```

## 任务队列约定

每个训练任务由 **toml + 同名目录** 组成：

| 内容 | 路径 | git |
|------|------|-----|
| 任务配置 | `tasks/<name>.toml` | ✅ 跟踪 |
| 训练集 | `tasks/<name>/` | ❌ 不跟踪（手动 rsync 到训练机） |

toml 中 `[[datasets.subsets]].image_dir` 应写成 `"./tasks/<name>"`（queue_runner 从仓库根目录启动 train.py，相对路径直接生效）。

### 执行顺序

按文件名字母序。需要控制顺序时用前缀：`001_xxx.toml` → `002_yyy.toml` → ...

### 任务状态

跑完后 `logs/done/` 会出现一个 marker 文件：

| 后缀 | 含义 |
|------|------|
| `.done` | 已执行（含 train.py 失败的，避免堵队列） |
| `.missing_dataset` | 同名训练集目录不存在或空，已跳过并发邮件 |

文件名带 sha8 内容指纹：`<name>__<sha8>.<suffix>`。**编辑 toml 内容会让 hash 变化，自动重新排队。** 想让一个失败任务重跑就改 toml 任意内容（加空格也行）。

## 配置

每个 toml 必须包含全路径：
- `dit_path`, `qwen3_path`, `vae_path`（基模，对应 `models/base/` 下文件）
- `output_dir`, `logging_dir`
- `[[datasets.subsets]].image_dir` → `"./tasks/<name>"`

## 部署

### 1. GitHub Secrets（8 个）

```
TRAIN_HOST       训练服务器 IP / 域名
TRAIN_USER       SSH 用户名
TRAIN_SSH_KEY    SSH 私钥
SMTP_HOST        SMTP 服务器
SMTP_PORT        SMTP 端口（一般 587）
SMTP_USER        SMTP 用户名 / 发件邮箱
SMTP_PASS        SMTP 密码 / 授权码
NOTIFY_EMAIL     收件邮箱
```

### 2. 训练服务器初始化

```bash
# clone 仓库
git clone <repo_url> ~/LoraCI
cd ~/LoraCI

# Python 环境
pip install -r requirements.txt --no-build-isolation
```

### 3. 填入基模下载 URL

编辑 `scripts/download_models.sh`，把三处 `REPLACE_ME` 改为真实的下载链接，或在训练机的 shell rc 里 `export ANIMA_BASE_URL=... QWEN3_URL=... VAE_URL=...`。

### 4. 上传训练集

每个任务上传到 `~/LoraCI/tasks/<name>/`：

```bash
rsync -av ./my_dataset/ <user>@<train_host>:~/LoraCI/tasks/001_topic/
```

### 5. 提交 toml 触发训练

```bash
git add tasks/001_topic.toml
git commit -m "queue topic training"
git push origin main
# → GitHub Actions 触发 → queue_runner 启动 → 跑完发邮件
```

## 邮件通知覆盖

| 失败位置 | 通知方式 |
|---|---|
| `git fetch/reset` 失败 | GitHub Actions 红叉 |
| 模型下载失败 | queue_runner crashed 邮件 |
| 训练集目录缺失 | dataset missing 邮件（任务跳过） |
| `train.py` 内任何阶段失败（含 import / argparse / config 加载 / 训练运行时） | training failed 邮件 + traceback |
| 训练成功 | training complete 邮件 |
| queue_runner 自身崩溃 | queue runner crashed 邮件 |

## 本地手动跑（无 CI）

```bash
python train.py --config_file tasks/001_topic.toml
```

train.py 仍然走完整的 try/except + 邮件通知（如果设了 SMTP 环境变量）。
