# Anima-Standalone-Trainer 部署手册

> 本手册按 2026-06-07 实际跑通的案例编写（maruko_v4，RTX 5090）。
> 2026-07-14 在第二个实例 `nI3WCI@wp08.unicorn.org.cn -p 19659` 二次验证通过（liu_v1，57 图），新踩坑见文末踩坑表与各节补充。

## 0. 环境（本次实测）

| 项目 | 值 |
|------|----|
| 入口 | `ssh -p 15864 dxpb3F@wp08.unicorn.org.cn`（已免密） |
| 容器 | Docker 容器，Ubuntu，passwordless sudo |
| GPU | RTX 5090 32GB，驱动 570.86.10（仅支持到 CUDA 12.8 / cu128） |
| Python | 系统 3.10.12，pip 25.1.1 |
| 资源 | 96 核 / 251GB RAM / ~600GB 可用盘 |
| 初始工具 | 只有 wget；无 git/node/conda |

**网络关键约束（决定了所有镜像选择）：**
- GitHub 可达但**不稳定**（大文件 tar 易被截断）→ 用 `ghfast.top` 镜像
- **HuggingFace 被墙** → 用 `hf-mirror.com`
- PyPI 直连慢 → 用清华源 `https://pypi.tuna.tsinghua.edu.cn/simple`

## 1. 部署仓库

```bash
cd ~
# 直连 codeload 通常可用；失败/截断则换 ghfast.top 镜像并校验 gzip
wget -q "https://codeload.github.com/gazingstars123/Anima-Standalone-Trainer/tar.gz/refs/heads/main" -O anima.tar.gz
gzip -t anima.tar.gz || wget -q "https://ghfast.top/https://github.com/gazingstars123/Anima-Standalone-Trainer/archive/refs/heads/main.tar.gz" -O anima.tar.gz
tar xzf anima.tar.gz && mv Anima-Standalone-Trainer-main Anima-Standalone-Trainer
```

## 2. 系统依赖 + 虚拟环境

```bash
# 容器缺 venv 模块/编译器/解压工具，一次装齐
sudo apt-get update -qq
sudo apt-get install -y -qq python3.10-venv python3-dev git \
    libgl1 libglib2.0-0          # cv2 运行需要 libGL.so.1

cd ~/Anima-Standalone-Trainer
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
```

## 3. 安装 Python 依赖

`requirements.txt` 已 pin `torch==2.7.0+cu128`，**正好匹配驱动 570**，且支持 RTX 5090 的 Blackwell（sm_120）。直接装即可：

```bash
# 注意：二次验证实例(19659) pytorch.org 直连被墙，requirements 里的 --extra-index-url 走不通 →
# 用阿里源作主 index（torch wheel 命中即用、cache 复用），并加 --retries/--timeout 防包文件间歇超时；
# 实测单轮即成(`Using cached ... +cu128`)。若仍卡，套 retry-loop（整轮失败重启）后台 nohup 跑。
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --retries 30 --timeout 60
# 验证 GPU 可用
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望: 2.7.0+cu128 True NVIDIA GeForce RTX 5090
```

## 4. 下载底模（hf-mirror.com）

三个文件来自 `circlestone-labs/Anima`，共约 5.6GB：

```bash
cd ~/Anima-Standalone-Trainer
mkdir -p models/dit models/te models/vae
B="https://hf-mirror.com/circlestone-labs/Anima/resolve/main/split_files"
wget -c "$B/diffusion_models/anima-base-v1.0.safetensors" -O models/dit/anima-base-v1.0.safetensors  # 4.18GB DiT
wget -c "$B/text_encoders/qwen_3_06b_base.safetensors"    -O models/te/qwen_3_06b_base.safetensors     # 1.19GB 文本编码器
wget -c "$B/vae/qwen_image_vae.safetensors"               -O models/vae/qwen_image_vae.safetensors      # 254MB VAE
```

下载后核对字节数（防截断）：
```
anima-base-v1.0.safetensors  4182218328
qwen_3_06b_base.safetensors  1192135096
qwen_image_vae.safetensors    253806246
```

## 5. 数据集

```bash
# 本地 maruko.zip 上传后解压（容器无 unzip，用 python）
scp -P 15864 ~/models/maruko.zip dxpb3F@wp08.unicorn.org.cn:~/Anima-Standalone-Trainer/jobs/maruko_v4/
ssh -p 15864 dxpb3F@wp08.unicorn.org.cn \
  'cd ~/Anima-Standalone-Trainer/jobs/maruko_v4 && python3 -c "import zipfile;zipfile.ZipFile(\"maruko.zip\").extractall(\".\")"'
```

数据集目录 `jobs/maruko_v4/maruko/`：同名 `NN.png` + `NN.txt`（Danbooru tag，逗号分隔，首词为触发词 `maruko`）。本次 83 图 + 83 标注。

## 6. 配置：拆成 config.toml + dataset.toml

训练脚本 `anima_train_network.py` 通过 `library/train_util.py::read_config_from_file` 读取**单个** `--config_file`：它把所有 section 扁平化合并进 argparse namespace，**多余的 key 不会报错**（如 `multigpu_mode`/`deepspeed`/各种 `fsdp_*` 会被忽略）。数据集则由 `dataset_arguments.dataset_config` 指向**另一个**文件。

所以把原始单文件 `maruko_v4.toml` 拆成两份，放在 `jobs/maruko_v4/`：

**config.toml**（训练主配置，路径必须用绝对路径）：
```toml
[model_arguments]
dit_path   = "/home/dxpb3F/Anima-Standalone-Trainer/models/dit/anima-base-v1.0.safetensors"
qwen3_path = "/home/dxpb3F/Anima-Standalone-Trainer/models/te/qwen_3_06b_base.safetensors"
vae_path   = "/home/dxpb3F/Anima-Standalone-Trainer/models/vae/qwen_image_vae.safetensors"

[dataset_arguments]
dataset_config = "/home/dxpb3F/Anima-Standalone-Trainer/jobs/maruko_v4/dataset.toml"
cache_latents_to_disk = true
cache_text_encoder_outputs_to_disk = false

[training_arguments]
output_name = "maruko_v4"
output_dir  = "/home/dxpb3F/Anima-Standalone-Trainer/jobs/maruko_v4/output"
logging_dir = "/home/dxpb3F/Anima-Standalone-Trainer/jobs/maruko_v4/logs"  # tensorboard 必填
save_model_as = "safetensors"
max_train_steps = 2400
save_every_n_steps = 600
learning_rate = 0.0001
text_encoder_lr = 5e-5
optimizer_type = "AdamW8bit"
lr_scheduler = "cosine"
mixed_precision = "bf16"
save_precision = "bf16"
gradient_checkpointing = true
torch_compile = true            # 用 inductor，需 gcc/g++（容器自带）
seed = 42

[network_arguments]
network_module = "networks.lora_anima"
network_dim = 48                # 见下方"LoRA 体积"
network_alpha = 48              # 保持 alpha=dim，scale 不变
network_train_unet_only = true

[anima_arguments]
timestep_sample_method = "logit_normal"
discrete_flow_shift = 3
weighting_scheme = "logit_normal"
```

**dataset.toml**：
```toml
[general]
enable_bucket = true
bucket_no_upscale = true
min_bucket_reso = 512
max_bucket_reso = 1536
bucket_reso_steps = 64

[[datasets]]
resolution = [1024, 1024]
batch_size = 4
caption_extension = ".txt"

[[datasets.subsets]]
image_dir = "/home/dxpb3F/Anima-Standalone-Trainer/jobs/maruko_v4/maruko"
num_repeats = 10
keep_tokens = 2
caption_dropout_rate = 0.1
caption_tag_dropout_rate = 0.1
```

### LoRA 体积控制（按本次实证）
体积随 `network_dim` **线性**增长。实测 `dim=32 → 138.7MB`。要 ~200MB 以上：

| dim/alpha | 输出体积 |
|-----------|----------|
| 32 | 138.7 MB |
| **48** | **~208 MB**（本次采用，恰好满足"约200MB以上"） |
| 64 | ~277 MB |

保持 `alpha=dim`，learning rate 等不变，训练动态与原配置一致。

## 7. 启动训练

单卡，`--mixed_precision bf16`；因为 `torch_compile=true` 需加 `--dynamo_backend inductor`：

```bash
cd ~/Anima-Standalone-Trainer
nohup bash -c "source venv/bin/activate && \
  python -m accelerate.commands.launch --num_cpu_threads_per_process 1 \
    --mixed_precision bf16 --dynamo_backend inductor \
    anima_train_network.py --config_file=jobs/maruko_v4/config.toml; \
  echo TRAIN_EXIT_\$?" > jobs/maruko_v4/train.log 2>&1 &
```

本次实测：2400 步 / 12 epoch / batch4@1024；前几步 torch_compile 预热较慢，稳定后约 4s/it；显存约 15GB/32GB；每 600 步存一次。

## 8. 查看进度

```bash
ssh -p 15864 dxpb3F@wp08.unicorn.org.cn \
  'cd ~/Anima-Standalone-Trainer; tail -c 400 jobs/maruko_v4/train.log | tr "\r" "\n" | tail -3; \
   ls -la jobs/maruko_v4/output/; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader'
```

## 9. 输出 + 下载到本地

```
jobs/maruko_v4/output/
├── maruko_v4.safetensors              # 最终 LoRA（~208MB）
└── maruko_v4-step0000{600,1200,1800,2400}.safetensors
```

```bash
scp -P 15864 dxpb3F@wp08.unicorn.org.cn:~/Anima-Standalone-Trainer/jobs/maruko_v4/output/maruko_v4.safetensors ~/models/
# 校验：本地与远程 sha256sum 一致
```

## 10. 踩坑记录（本次实际遇到）

| 现象 | 原因 | 解决 |
|------|------|------|
| `ImportError: libGL.so.1` | cv2 缺系统库 | `sudo apt install libgl1 libglib2.0-0` |
| `ensurepip is not available` | 容器无 venv 模块 | `sudo apt install python3.10-venv` 后重建 venv |
| github tar 解压报 `unexpected EOF` | 直连被截断 | `gzip -t` 校验；改 `ghfast.top` 镜像 |
| HF 下载无响应 | HuggingFace 被墙 | 全部换 `hf-mirror.com` |
| 数据集解压失败 | 容器无 `unzip` | 用 `python3 -c "import zipfile;..."` |
| nohup 里 `~` 不展开 | 非交互 shell | 配置内全用绝对路径 `/home/dxpb3F/...` |
| `pip install -r` 直连卡死 40min+ 不动 | `download.pytorch.org/whl/cu128` 直连被墙；且各 PyPI 镜像**包文件**下载间歇超时（index 页通、wheel 随机卡） | 改用阿里源 `-i https://mirrors.aliyun.com/pypi/simple/ --retries 30 --timeout 60`，并套 retry-loop（整轮失败重启）；torch wheel 一旦下进 cache 即 `Using cached` 复用，不再走 pytorch.org |
| `bubu_v1.toml` 单文件 `dataset_config=""` 跑不通 | `dataset_config` 被当数据集配置文件加载，`""`→`"."` → file not found；改成指向自身后 voluptuous 把整个文件喂给数据集 schema，`model_arguments` 等被报 `extra keys not allowed` | **必须按第 6 节拆成两文件**：主 config（`model/training/network/anima_arguments`，多余 key 被扁平化忽略）+ 独立 `dataset.toml`（只含 `[general]`/`[[datasets]]`），`dataset_config` 指向后者。`bubu_v1.toml` 那种单文件内联是跑不通的模板遗留 |
| GitHub 仓库直连即可 | 两次实例 `codeload.github.com/<repo>/tar.gz/refs/heads/main` 均直连成功且 gzip 校验通过，无需 ghfast.top | 优先 codeload 直连；失败再回退 ghfast.top |
