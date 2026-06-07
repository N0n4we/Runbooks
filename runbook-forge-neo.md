# Forge-Neo 部署手册

> 本手册按 2026-06-07 实际跑通的案例编写（RTX 5090，自签 HTTPS，SSH 隧道访问）。
> 上游：`Haoming02/sd-webui-forge-classic` 的 `neo` 分支（`6Morpheus6/forge-neo` 只是 Pinokio 壳，无头服务器不用）。

## 0. 环境

| 项目 | 值 |
|------|----|
| 入口 | `ssh -p 15864 dxpb3F@wp08.unicorn.org.cn`（免密） |
| GPU | RTX 5090 32GB，驱动 570.86.10 → **只支持到 CUDA 12.8（cu128）** |
| Python | uv 托管 3.13.13（推荐）；系统为 3.10.12 |
| 部署路径 | `~/sd-webui-forge-neo/` |

**网络约束同 trainer：** GitHub 不稳→`ghfast.top`；HF 被墙→`hf-mirror.com`；PyPI 慢→清华源。

## 头号坑：默认 PyTorch 是 cu130，本机驱动装不了
`modules/launch_utils.py` 默认装 `torch==2.11.0+cu130`，并有硬性驱动检查，驱动不够会直接报
`Please update your GPU driver to support cu130`。**必须用 `TORCH_COMMAND` 覆盖成 cu128**（见第 3 步 webui-user.sh）。

## 1. 部署仓库（ghfast.top 镜像，直连易截断）

```bash
cd ~
for M in "https://ghfast.top/https://github.com/Haoming02/sd-webui-forge-classic/archive/refs/heads/neo.tar.gz" \
         "https://gh-proxy.com/https://github.com/Haoming02/sd-webui-forge-classic/archive/refs/heads/neo.tar.gz"; do
  wget -q --tries=1 "$M" -O forge.tar.gz && gzip -t forge.tar.gz && break
done
tar xzf forge.tar.gz && mv sd-webui-forge-classic-neo sd-webui-forge-neo
```

## 2. 系统依赖 + uv 虚拟环境（Python 3.13）

```bash
sudo apt-get update -qq
sudo apt-get install -y -qq python3.10-venv python3-dev git libgl1 libglib2.0-0
pip install -q uv                      # 落在 ~/.local/bin
cd ~/sd-webui-forge-neo
~/.local/bin/uv venv venv --python 3.13 --seed   # 自动下载托管 CPython 3.13
```

## 3. 自签证书 + webui-user.sh

```bash
cd ~/sd-webui-forge-neo && mkdir -p certs
openssl req -x509 -newkey rsa:2048 -nodes -keyout certs/key.pem -out certs/cert.pem \
  -days 3650 -subj "/CN=localhost" -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"
```

**`~/sd-webui-forge-neo/webui-user.sh`**（这是把所有坑一次性解决的核心文件）：
```bash
#!/usr/bin/env bash
export PATH="$HOME/.local/bin:$PATH"          # uv 在 PATH 上，webui.sh 才会用 uv

# 关键：覆盖默认 cu130 → cu128（匹配驱动 570、支持 5090）
export TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
export TORCH_COMMAND="pip install torch==2.8.0 torchvision==0.23.0 --extra-index-url https://download.pytorch.org/whl/cu128"

# HF 被墙
export HF_ENDPOINT="https://hf-mirror.com"

# PyPI 直连/Fastly 限速 → 清华源（torch 仍走上面的 pytorch.org）
export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
export UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"

# 自签 HTTPS，绑 127.0.0.1（不加 --listen → 只监听 localhost，正好走 SSH 隧道）
export COMMANDLINE_ARGS="--uv --port 7860 \
  --tls-keyfile $HOME/sd-webui-forge-neo/certs/key.pem \
  --tls-certfile $HOME/sd-webui-forge-neo/certs/cert.pem"

exec "$(dirname "$0")/webui.sh"
```
```bash
chmod +x webui.sh webui-user.sh
```

说明：不加 `--listen` 时 `gradio_server_name()` 返回 `None`，服务绑定 `127.0.0.1`，更安全且匹配隧道；TLS 由 forge 进程内的 uvicorn 终结。

## 4. 首次启动（会装 torch+依赖，耗时长）

```bash
cd ~/sd-webui-forge-neo
nohup ./webui-user.sh > launch.log 2>&1 &     # 用最简单的 nohup；勿用 setsid（曾导致 ssh 255）
```
等到日志出现 `Running with TLS` + `Running on local URL: https://127.0.0.1:7860` 即成功。
torch 2.8.0+cu128 走 pytorch.org（Fastly）较慢，耐心等；其余依赖走清华源很快。

## 5. 预放 ESRGAN（否则首次放大/hires 会阻塞队列）

forge 首次用到上采样器时会从 `github.com/cszn/KAIR/...` 下 `ESRGAN.pth`（66.9MB）。这个下载在**生成流程内同步进行**，国际线路只有 ~15KB/s，会把整个队列卡住。提前放好：

```bash
cd ~/sd-webui-forge-neo/models/ESRGAN
wget -q "https://ghfast.top/https://github.com/cszn/KAIR/releases/download/v1.0/ESRGAN.pth" -O ESRGAN.pth
# 校验 66929193 字节；forge 的 load_file_from_url 见文件存在即跳过下载
```
> 同理，人脸修复（facexlib/GFPGAN）等首次也会拉 github，必要时用同样方式预放到对应目录。

## 6. 挂载模型（软链接，不复制）

模型与训练目录在同一文件系统、盘已较满，用软链接接入（forge 启动时扫描，加完需**重启** forge 才会出现在下拉框）：

```bash
cd ~/sd-webui-forge-neo/models; SRC=~/Anima-Standalone-Trainer
ln -sf "$SRC/models/dit/anima-base-v1.0.safetensors"  Stable-diffusion/anima-base-v1.0.safetensors
ln -sf "$SRC/models/te/qwen_3_06b_base.safetensors"   text_encoder/qwen_3_06b_base.safetensors
ln -sf "$SRC/models/vae/qwen_image_vae.safetensors"   VAE/qwen_image_vae.safetensors
ln -sf "$SRC/jobs/maruko_v4/output/maruko_v4.safetensors" Lora/maruko_v4.safetensors
```

目录对应：DiT→`Stable-diffusion/`，文本编码器→`text_encoder/`，VAE→`VAE/`，LoRA→`Lora/`。
验证已识别：`GET https://127.0.0.1:7860/config` 里能搜到模型名即可。

## 7. 访问 / 重启 / 停止

```bash
# 本地建隧道（127.0.0.1，因为 forge 只绑 localhost）
ssh -N -L 7860:127.0.0.1:7860 -p 15864 dxpb3F@wp08.unicorn.org.cn
# 浏览器：https://localhost:7860 （自签证书点一次"继续"）

# 重启（依赖已装，秒级启动）
ssh -p 15864 dxpb3F@wp08.unicorn.org.cn \
  'cd ~/sd-webui-forge-neo && pkill -f launch.py; sleep 2; nohup ./webui-user.sh > launch.log 2>&1 &'

# 停止
ssh -p 15864 dxpb3F@wp08.unicorn.org.cn 'pkill -f launch.py'
```

## 8. 踩坑记录（本次实际遇到）

| 现象 | 原因 | 解决 |
|------|------|------|
| `update your GPU driver to support cu130` | 默认 torch=cu130，驱动 570 不够 | `TORCH_COMMAND` 改 cu128（torch 2.8.0） |
| gradio/依赖装得极慢 | PyPI/Fastly 国际限速 | `UV_*_INDEX`/`PIP_INDEX_URL` 指清华源 |
| 队列卡死、GPU 0% | 生成中同步下 `ESRGAN.pth`（~15KB/s） | 预放 ESRGAN.pth（第 5 步） |
| `ImportError: libGL.so.1` | cv2 缺库 | `apt install libgl1 libglib2.0-0` |
| ssh 启动命令返回 255、进程没起 | setsid/pkill 同句牵连会话+线路抖动 | 用最简 `nohup ./webui-user.sh &`，启动与 kill 分开执行 |
| `pgrep -f launch.py` 误报运行中 | 匹配到自己命令行里的 "launch.py" | 改判断 HTTPS 是否 200 / 看 launch.log mtime |
| github tar 截断 | 直连不稳 | `ghfast.top` + `gzip -t` 校验 |
