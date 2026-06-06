# LoraCLI

Anima LoRA 训练命令行工具。在自己的 GPU 机器上手动跑单个训练任务。

> 训练代码迁移自 `github.com/gazingstars123/Anima-Standalone-Trainer`

## 用法

```bash
python train.py --config_file tasks/001_topic.toml
```

`train.py` 是唯一入口：加载 toml 配置 → 运行训练。整个过程包在
`try/except` 里，结束时如果配置了 SMTP 环境变量会发一封成功/失败通知邮件，
没配置就静默跳过（不影响训练）。

## 仓库结构

```
train.py                # 训练入口（CLI）
anima_train_network.py  # AnimaNetworkTrainer
train_network.py        # NetworkTrainer base
library/                # 训练库
networks/               # LoRA 实现
configs/                # tokenizer 配置
scripts/
  download_models.sh    # 基模下载助手（需先填入真实 URL）
tasks/                  # 训练任务（toml + 同名数据集目录）
models/base/            # 基模存放处
logs/                   # 训练日志、输出
```

## 准备

1. 下载基模到 `models/base/`：

   ```bash
   # 先在 scripts/download_models.sh 里把 REPLACE_ME 换成真实 URL，
   # 或通过环境变量传入：
   ANIMA_BASE_URL=... QWEN3_URL=... VAE_URL=... \
     bash scripts/download_models.sh models/base
   ```

2. 准备任务：把配置写成 `tasks/<name>.toml`，数据集放在同名目录
   `tasks/<name>/`（数据集本身不纳入版本控制，见 `.gitignore`）。

3. 运行：

   ```bash
   python train.py --config_file tasks/<name>.toml
   ```

## 可选：邮件通知

训练结束后可发一封成功/失败邮件。仅当下列环境变量齐全时生效，
否则静默跳过：

```bash
export SMTP_HOST=smtp.example.com
export SMTP_PORT=587          # 可选，默认 587（STARTTLS）
export SMTP_USER=you@example.com
export SMTP_PASS=app-password
export NOTIFY_EMAIL=you@example.com
```
