# MINIMAX-H3：部署与双采样运行手册

> **当前状态（2026-11-01）**：H3 在 `ssh -p 14499 root@connect.weste.seetacloud.com` 上可用，
> 六个权重齐全（`/root/autodl-tmp/h3-replace/models`，43 GiB），ComfyUI 端口 8188。
> 该机同时部署 Fibo2 / Qwen / H3 三套权重，共用 `service.lock`，同一时间只能跑一个服务：
> 跑 H3 前先停 Fibo2（`kill -TERM <run-fibo2.py pid>`），再 `bash /root/h3-replace/start-comfy.sh`。
> 2026-11-01 已在该机跑通 240 秒人物的链式替换（24 段），见文末「链式人物替换实测」。

> **历史验收服务器（2026-10-31，清理前）**：`ssh -p 30433 root@connect.nmb2.seetacloud.com`，
> RTX 3090 24GB；已完成部署、模型校验及 **10 秒/480p 双采样实测**。
> 新机命令、分阶段耗时/显存及磁盘瓶颈验证见文末章节。下述 4090D 实测属于已关闭的旧服务器，
> 不代表新机的性能或显存验收；不要再连接旧端口 48995。

> 2026-10-30 已完成真实 GPU 验收：同一个请求输入三张图、10 秒视频和提示词，
> 一采、二采均输出带声音的 MP4。模型、素材及输出均保留在服务器，没有经本机中转。

## 本机 `run.py`：自动 SSH 上传与取回双采样视频

**须先恢复已清理的两个权重和 ComfyUI 服务**。包装脚本会在上传之前检查六个模型和服务，不会自动下载权重、启动服务或重复提交；目前运行会明确报告缺失模型。恢复 H3 时避免与 Fibo2 同时推理，两者共享同一张 24 GiB GPU 和容器内存。

```bash
cd /Users/yingtiankai/Desktop/Runbooks/video-generation/h3-replace
uv run --no-project python run.py \
  --images /path/1.png /path/2.png /path/3.png \
  --video /path/reference.mp4 \
  --prompt 'Use <Picture 1>, <Picture 2>, <Picture 3> as references. Follow the motion of <Video 1>.' \
  --seconds 10 --seed 42 --output-dir ./h3-result
```

本机只需 Python 标准库、系统 `ssh`/`scp`、可用的 SSH 密钥，以及同目录的 `run-h3.py`；不需要本机 Torch、ComfyUI、模型或手动建隧道。默认连接 `ssh -p 30433 root@connect.nmb2.seetacloud.com`；迁移时修改 `run.py` 顶部部署常量。两个项目的 `run.py` 开头均有完整用法注释。

- 恰好三张图片（480 或 720 像素高）、一个 2–15 秒参考视频及提示词；尺寸/时长由服务器原脚本 ffprobe 验证。`--seconds` 指生成时长，默认 10 秒，对齐为 243 帧/24fps。
- 自动 SCP 上传本地媒体，SSH 调用现有服务器脚本，在**一个 ComfyUI 请求**里保留一采、二采；不修改原始工作流或服务器 `run-h3.py`。
- 本机输出 `h3-result/pass1.mp4`（576×320）、`pass2.mp4`（864×480）、`result.json`、`run.log`。两份视频都含音轨，下载后逐个比对服务器 SHA256；已有输出目录时拒绝覆盖。
- 服务器原任务、ComfyUI 输入和成片均保留，成功后只删除本次上传中转目录。失败保留中转目录及日志；远端任务使用 setsid/nohup，断线后可能仍在执行，先检查打印目录下的 `run.log` 和任务记录，**不要盲目重跑**。
- 测试命令：`uv run --no-project python -B -m unittest -v test_run`。

包装层已通过 6 项单元测试；另实测 SSH 缺模型预检正确拦截、未上传输入/创建输出目录，并通过真实 SSH/SCP 下载和 SHA256/ffprobe 校验历史任务 `h3_20261031_125525_013d907a` 的两份成片。证据在 `tmp-h3-wrapper-20261031.8H6hNf/`。**这是历史成片传输验收，没有在缺权重状态下新跑 H3 推理，也没有恢复模型或启动服务。**

## 范围（以下保留历史部署与验收记录）

- SSH：`ssh -p 48995 root@connect.cqa1.seetacloud.com`。
- 保留原始 `MINIMAX-H3.json`，`run-h3.py` 在服务器生成适配后的 UI/API JSON。
- 模型和公开素材全部由服务器直接下载，不经过本机中转。
- 一次 API 请求同时输入 3 张 480p 图片、1 段 10 秒参考视频和提示词，保留一采、二采两份有声 MP4。
- 当前采用用户确认的 FL2VA pruned INT8 主模型和 NVFP4 AWQ 编码器，不使用原 JSON 下拉框里的 hybrid/INT8 encoder。

## 已确认的环境

| 项目 | 实测 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 D，24564 MiB |
| 驱动 | 595.71.05 |
| 基础 Python | `/root/miniconda3/bin/python`，3.12.3 |
| PyTorch | `2.8.0+cu128`，CUDA 12.8，可正常使用 GPU |
| torchvision / torchaudio | `0.23.0+cu128` / `2.8.0+cu128` |
| CUDA 编译器 | `/usr/local/cuda/bin/nvcc`，12.8 |
| Python 环境 | uv 创建 `/root/h3-venv`，继承基础环境，避免复制/升级 CUDA torch |
| 数据盘 | `/root/autodl-tmp`，50 GiB；六个模型约 45.12 GB（十进制） |
| 容器内存限制 | 约 62 GiB，不是 `free -h` 显示的宿主机 503 GiB |

## 文件布局

```text
/root/h3-replace/                  # 此目录的 JSON、脚本、部署日志和每次任务证据
  MINIMAX-H3.json
  run-h3.py
  constraints.txt
  models.tsv                      # 文件名、预期字节数、SHA256、直下镜像 URL
  download-models.sh
  log/
  fixtures/
  jobs/
/root/h3-venv/
/root/comfy/ComfyUI/
/root/autodl-tmp/h3-replace/models/ # 模型放数据盘，不占满系统盘
```

## 网络预检与安装原则

本次 Hugging Face 直连超时；ModelScope 和 hf-mirror 的 1 MiB 范围请求均成功，
ModelScope 更快，因此模型使用 ModelScope。GitHub codeload 直连约 40 KB/s，
已终止并改为 `gh-proxy.com` 源码压缩包。没有使用 git 命令。

先验证基础 torch，再用 uv 创建 `--system-site-packages` 环境。
依赖安装必须保留约束：`torch==2.8.0`、`torchvision==0.23.0`、
`torchaudio==2.8.0`、`triton==3.4.0`。uv 的依赖解析会尝试复制继承的 CUDA 包，
因此 torchaudio 使用 `uv pip install --no-deps`，其余依赖在该 uv 环境内使用 pip
解析，以复用已验证的系统 torch。

```bash
export PATH=/root/h3-venv/bin:/root/.local/bin:/usr/local/cuda/bin:$PATH
python -c 'import torch,torchaudio; print(torch.__version__,torch.version.cuda,torch.cuda.is_available(),torch.cuda.get_device_name(0),torchaudio.__version__)'
```

## 软件与节点版本

| 组件 | 本次版本 / 来源 |
|---|---|
| comfy-cli | `1.20.0`，已执行 `comfy tracking disable` |
| ComfyUI | `0.34.0`，使用源码压缩包，不需要 git |
| 前端 | `comfyui-frontend-package==1.49.6` |
| comfy-kitchen / aimdo | `0.2.31` / `0.4.15` |
| Triton | `3.4.0`，显式启用 kitchen Triton 后端 |
| transformers | `5.22.0` |
| KJNodes | Registry `1.5.0`，提供 FFN / LowVRAMAttention |
| VideoHelperSuite | Registry `1.7.9`，负责读参考视频及封装输出 |
| YCNodes MiniMax H3 | `146207295c1d2bd6b64a6552484cb6ed8be1c9ef`，提供 `H3SigmaRefiner` |
| LBH latent upscaler 节点 | `40316cf008b2fd8663263270669eb4da23f89d2c` |

首次安装的关键命令（**不要在已运行的部署上覆盖目录**）：

```bash
export PATH=/root/miniconda3/bin:/root/.local/bin:$PATH
python -m pip install --user uv -i https://mirrors.aliyun.com/pypi/simple/ --retries 30 --timeout 60
uv venv --system-site-packages --python /root/miniconda3/bin/python /root/h3-venv
uv pip install --python /root/h3-venv/bin/python --no-deps \
  --index-url https://mirrors.aliyun.com/pypi/simple/ torchaudio==2.8.0 pip
export PATH=/root/h3-venv/bin:/root/.local/bin:/usr/local/cuda/bin:$PATH

mkdir -p /root/h3-replace/log /root/comfy
printf 'torch==2.8.0\ntorchvision==0.23.0\ntorchaudio==2.8.0\ntriton==3.4.0\n' \
  >/root/h3-replace/constraints.txt
apt-get update
apt-get install -y --no-install-recommends ffmpeg aria2 libgl1 libglib2.0-0

test ! -e /root/comfy/ComfyUI
curl -fL --connect-timeout 10 --max-time 180 \
  -o /root/h3-replace/comfyui-v0.34.0.tar.gz \
  https://gh-proxy.com/https://github.com/Comfy-Org/ComfyUI/archive/refs/tags/v0.34.0.tar.gz
tar -xzf /root/h3-replace/comfyui-v0.34.0.tar.gz -C /root/comfy
mv /root/comfy/ComfyUI-0.34.0 /root/comfy/ComfyUI

# 不下载本次不需要的模板媒体包；前端和推理依赖仍安装。
awk '!/^comfyui-workflow-templates==/' /root/comfy/ComfyUI/requirements.txt \
  >/root/h3-replace/requirements-runtime.txt
timeout 600 python -m pip install \
  -i https://mirrors.aliyun.com/pypi/simple/ --retries 30 --timeout 60 \
  -c /root/h3-replace/constraints.txt \
  -r /root/h3-replace/requirements-runtime.txt comfy-cli==1.20.0
timeout 600 uv pip install --python /root/h3-venv/bin/python \
  --index-url https://mirrors.aliyun.com/pypi/simple/ \
  --constraint /root/h3-replace/constraints.txt \
  matplotlib color-matcher mss opencv-python-headless ninja
comfy tracking disable
```

长安装命令超时后可重跑，利用缓存；本机已保留 `install-python.sh`、
`install-node-deps.sh` 和各自日志。不要因 cu130 警告直接升级用户指定的 torch。
在此 `cu128` 环境中，kitchen CUDA 后端被自动禁用，INT8 使用 Triton，
NVFP4 使用兼容实现；已用真实任务验证，不要求 Blackwell GPU。

节点源码直接在服务器获取，解压到 `custom_nodes/`，或链接到
`/root/h3-replace/node-sources/` 下对应源码目录：

- [KJNodes 1.5.0 ZIP](https://cdn.comfy.org/kijai/comfyui-kjnodes/1.5.0/node.zip)
- [VideoHelperSuite 1.7.9 ZIP](https://cdn.comfy.org/kosinkadink/comfyui-videohelpersuite/1.7.9/node.zip)
- [YCNodes 固定修订](https://github.com/yichengup/ComfyUI-YCNodes-MiniMax-H3/tree/146207295c1d2bd6b64a6552484cb6ed8be1c9ef)
- [Upscaler 固定修订](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/tree/40316cf008b2fd8663263270669eb4da23f89d2c)

Upscaler 完整源码包带示例视频，网络慢时无需下载它们。此次使用 GitHub API
读取固定修订的文件树，再通过每个文件的 blob URL/base64 获取 `.py` 文件；
根 `__init__.py` 和 `nodes/` 下四个 `.py` 均已落盘。

## 模型下载与完整性

下面路径相对于 `/root/comfy/ComfyUI/models/`；对应目录已链接到数据盘。
全部文件经过**精确字节数、SHA256、safetensors 头**三重检查。

| 文件 | 字节数 | 张量数 |
|---|---:|---:|
| `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 20970379616 | 932 |
| `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 15687142551 | 2054 |
| `vae/minimax_h3_video_vae_fp16.safetensors` | 5207808496 | 562 |
| `vae/minimax_h3_audio_vae_fp32.safetensors` | 605254808 | 917 |
| `loras/minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors` | 1956192992 | 624 |
| `latent_upscale_models/minimax_h3_latent_upscaler_3d_fp16.safetensors` | 690592672 | 322 |

下载来源：

- 前四项：`https://modelscope.cn/models/Comfy-Org/MiniMax-H3/resolve/master/` + 表内路径。
- LoRA：`https://modelscope.cn/models/lightx2v/Minimax-h3-Turbo/resolve/master/minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors`。
- 放大模型：`https://modelscope.cn/models/LBH-123-AI/Minimax_h3_latent_Upscaler/resolve/master/minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors`。
  下载后使用表内旧文件名；新版名称和旧版 checkpoint 的 SHA256 完全相同。

SHA256（按上表顺序）：

```text
e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a
35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6
7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522
8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48
449d80f301ac571622c72e28b8fd72a4b3681b7a8df8a92f17c8f6ec43f56558
043e5a48e161610ef6c3ea974645220354d06fa618abca15f76d084812eb55c2
```

服务器 `models.tsv` 保存每一项的路径、字节数、SHA256、完整 URL。
`models.aria2` 是由此生成的下载清单；`download-models.sh` 使用
`--continue=true --check-integrity=true --max-concurrent-downloads=3`
和每文件 `--split=16 --max-connection-per-server=16`，不重复缓存整份模型。
下载完成后再次执行 `sha256sum -c`；成功标记为 `models.complete`。

```bash
# 断线后先确认没有同一下载进程；不要对同一文件并发启动两个 aria2。
pgrep -af aria2c
cd /root/h3-replace
setsid nohup bash download-models.sh >log/models.log 2>&1 </dev/null &
tail -f log/models.log
df -h / /root/autodl-tmp
```

本次完成后系统盘余约 23 GiB、模型盘余约 8 GiB，无需扩容。
**8 GiB 不足再并存一份约 21 GB 的 DiT；如后续要增加主模型，先扩容，不删除已验收模型。**

## 原图的运行时适配

`run-h3.py` 只适配本目录这一份导出，节点编号变化时需要更新映射；不是另写一套生成管线。

- 开启参考图像 1–3 和参考视频 1，关闭的音频及其余参考不接入 API 图。
- 三个活跃图像预处理改用 core `ImageScale`，在本次横向 480p 验收画布中心裁剪/缩放；
  不安装 LayerStyle 的无关依赖。其他宽高比不属于此次保真验收范围。
- KJ `SetNode/GetNode` 由 comfy-cli 解析为实连；剔除 rgthree、Note 和图像预览等非推理节点。
- `#114` Sage 补丁旁路，使用原生 PyTorch attention；`#416` 额外模型预览旁路。
- `#412` FFN `chunks=8, seq_threshold=4096`；`#413` `head_chunks=4`，不再是原图的 `1`。
- 一采 `576×320`，二采 latent 放大到 `0.4 MP`，实测为 `864×480`。
- 放大节点开启 `enable_temporal_chunking` 和 `force_unload`，CUDA / FP16 / align 32。
- 保留原图 Euler / simple / 基础 10 步、SigmaRefiner extra_steps=1、SplitSigmas step=5；
  实际一采 5 步、二采 6 步，LoRA strength=0.75，video/audio shift=6/3。
- 两个 VideoCombine 都设置 `save_output=true`，使用不同文件前缀；不是只有临时一采预览。
- 生成时长 10 秒经原图 `17n+5` 对齐得到 243 帧，即 24fps 下 10.125 秒。

原始 JSON 不改写；每次任务保存 UI 副本、实际 API 图、输入 ffprobe、提交响应、history 和结果。

## 公开验收素材

来源：[Blender Foundation《Sintel》公开预告片，W3C 托管](https://media.w3.org/2010/05/sintel/trailer.mp4)。
本次截取原片第 30–40 秒，避免较早片段的黑场；三张图取自该片段的第 1、4、8 秒，
是三张不同的帧，不是重复文件。`fixtures/SHA256SUMS` 保存了素材校验值。

```bash
mkdir -p /root/h3-replace/fixtures
cd /root/h3-replace/fixtures
aria2c -c -x16 -s16 -k1M --file-allocation=none --connect-timeout=10 \
  --timeout=30 --max-tries=3 --out=sintel-trailer.mp4 \
  https://media.w3.org/2010/05/sintel/trailer.mp4
ffmpeg -nostdin -y -v error -ss 30 -i sintel-trailer.mp4 -t 10 \
  -vf 'scale=864:480:force_original_aspect_ratio=increase,crop=864:480,setsar=1,fps=24' \
  -an -c:v libx264 -crf 19 -pix_fmt yuv420p reference_10s_480p.mp4
for pair in 1:1 2:4 3:8; do
  ffmpeg -nostdin -y -v error -ss "${pair#*:}" -i reference_10s_480p.mp4 \
    -frames:v 1 "reference_${pair%:*}_480p.png"
done
```

输入验证：三张 PNG 都为 `864×480`，文件哈希不同，亮度/对比度检查排除全黑图；
视频为 `864×480`、24fps、240 帧、精确 10.000 秒。
视频完整装入节点，没有人为抽成一张图；H3 内部会按原生规则向下对齐参考帧数，
240 帧会取 226 帧，Qwen 的视频理解分支内部再以 2fps 取样。这些是 core 的处理，不是脚本截短输入。

## 启动与运行

仅需从本机上传小配置/代码，不上传模型或素材：

```bash
scp -P 48995 MINIMAX-H3.json run-h3.py root@connect.cqa1.seetacloud.com:/root/h3-replace/
ssh -p 48995 root@connect.cqa1.seetacloud.com
export PATH=/root/h3-venv/bin:/root/.local/bin:/usr/local/cuda/bin:$PATH
cd /root/comfy/ComfyUI
setsid nohup python main.py --listen 127.0.0.1 --port 8188 \
  --disable-auto-launch --disable-api-nodes --reserve-vram 4 --enable-triton-backend \
  >/root/h3-replace/log/comfy.log 2>&1 </dev/null &
echo $! >/root/h3-replace/comfy.pid
curl -fsS --max-time 15 http://127.0.0.1:8188/system_stats
```

服务已运行时不要重复启动。重启前确认队列为空，并且 PID 对应本次的 `main.py`，
再终止 `comfy.pid` 记录的进程；不要使用全局 `pkill python`。
所有长下载、服务和推理都使用 nohup/setsid，SSH 断开不会停止任务。

`run-h3.py` **在目标服务器内运行**，参数是服务器文件路径；不自动 SSH、不上传或下载媒体。
脚本使用 comfy-cli 转换原始工作流，再以单个 ComfyUI API 请求运行两个采样分支。
会验证输入尺寸/参考视频时长，并用服务器 ffprobe 检查两份输出的尺寸、帧数、帧率、
H.264 视频及 AAC 音轨；ffprobe 不导出 MP4 中庞大的工作流 comment 元数据。

```bash
export PATH=/root/h3-venv/bin:/root/.local/bin:/usr/local/cuda/bin:$PATH
cd /root/h3-replace
python run-h3.py --self-test

python run-h3.py \
  --images fixtures/reference_1_480p.png fixtures/reference_2_480p.png fixtures/reference_3_480p.png \
  --video fixtures/reference_10s_480p.mp4 \
  --prompt 'Create one continuous 10-second cinematic animation shot. Use <Picture 1>, <Picture 2> and <Picture 3> as visual references for the subject, environment and lighting. Follow the motion and camera style of <Video 1>. Preserve identity and scene continuity, with natural movement, subtle ambient wind and footsteps. No subtitles, no text overlays, no voice-over.' \
  --seconds 10 --seed 42

# 同一命令加 --dry-run 只验证素材并转换，不提交 GPU 推理。
# 已有 acceptance.sh 包含完整验收命令，可在断线安全模式下运行：
setsid nohup bash acceptance.sh >log/acceptance.log 2>&1 </dev/null &
```

两条运行示例是**二选一**，不是同时提交。脚本不会自动重复提交失败任务。
三个参考图需要 480 或 720 像素高；参考视频按本图约束为 2–15 秒。
只有帧数数学自检覆盖 20 秒，**没有验证 20 秒参考视频或生成任务**。

## 实测验收

- 单一 prompt ID：`4f41783a-2f73-4283-b355-bc23f68f2f17`。
- run ID：`h3_20261030_211944_6dea050f`。
- ComfyUI history：`completed=true, status_str=success`，同时包含输出节点 `408` 和 `299`。
- 日志确认一采 5 步、二采 6 步；208 个 LoRA patches 生效；3D upscaler 实际执行并卸载回 CPU。
- 后端执行用时 `680.995` 秒，约 **11 分 21 秒**，不含模型下载和安装。

| 输出 | 一采 | 二采 |
|---|---|---|
| 文件 | `pass1_00001-audio.mp4` | `pass2_00001-audio.mp4` |
| 尺寸 | `576×320` | `864×480` |
| 视频 | H.264，243 帧，24fps，10.125 秒 | H.264，243 帧，24fps，10.125 秒 |
| 音轨 | AAC，32000Hz，双声道 | AAC，32000Hz，双声道 |
| 文件字节数 | 1109211 | 1739824 |
| 采样亮度均值 / 标准差 | 92.507 / 67.525 | 95.063 / 69.529 |
| 2fps 灰度帧间 MAD | 17.635 | 19.109 |
| 平均音量 | -40.7 dB | -32.1 dB |

两份成片均已排除全黑、静帧和完全静音；这不等同于人工审片或参考一致性评分。

输出目录：

```text
/root/comfy/ComfyUI/output/h3_20261030_211944_6dea050f/
```

目录同时有不带音轨的中间 `.mp4` 和带声音的 `-audio.mp4`，交付使用后者。
`jobs/<run_id>/verification.json` 保存精简 ffprobe、画面/音频统计和硬件采样结果，
`history.json`、`workflow.api.json` 是单任务双输出的审计依据。

资源实测（5 秒采样，不是 CUDA allocator 的精确瞬时峰值）：

- GPU 峰值 **19940 MiB，约 19.47 GiB**；24GB 4090D 全流程通过。
- 容器内存采样峰值 **57.576 GiB**，包括文件缓存，不应解读为纯张量内存硬下限。
- cgroup `oom=0, oom_kill=0`；未发生 GPU OOM。
- 只有此低分辨率一采 + 480p 二采档完成验证，不能推广到原图 1.5MP / 15 秒或 720p。

## 已知边界与排障

- 这是 CLI/API 跑通部署，不是原始 GUI 所有插件的全功能安装；未安装 LayerStyle、rgthree、
  Sage 编译内核和模板媒体包。模板包缺失会在启动日志提示，但服务和本图 API 执行正常。
- 不把旧 `widgets_values` 的 hybrid/INT8 encoder 当作实际模型；本次明确使用用户指定的
  FL2VA pruned + NVFP4 AWQ。采用 Ref2VA 节点完成了技术执行，不代表已做语义参考质量评测。
- 首次计算会有 Triton 编译 / DynamicVRAM 初始化；不要因短时没有采样百分比就重复提交。
- CUDA 后端的 cu130 警告在此环境是预期的；保留 torch 2.8.0，确认 Triton 已启用。
- 内存需看 `/sys/fs/cgroup/memory.current` 和 `memory.events`，不是宿主机 `free -h`。
  此内核没有 `memory.peak`；峰值取日志的采样最大值。
- 增加分辨率、视频时长、参考数量后需重新测试显存/RAM，不能承诺仍可用当前资源运行。
- 视频按原图设置带 prompt / 工作流元数据，分享成片前注意其中的提示词和路径信息。
- 本次没有将生成视频下载到本机，也没有进行人工视觉/听觉审片。

## 新服务器部署（2026-10-31，仅部署不推理）

本节对应用户新开的服务器，只完成运行 `run-h3.py` 前的准备；
没有提交 `/prompt`，没有执行旧机的 `acceptance.sh`，没有生成新视频。
原始 `MINIMAX-H3.json` 和 `run-h3.py` 均保持原样，本地与服务器 SHA256 一致。

### 环境与安装结果

| 项目 | 新机实测 |
|---|---|
| SSH | `ssh -p 30433 root@connect.nmb2.seetacloud.com` |
| GPU / 驱动 | RTX 3090，24576 MiB，sm_86 / 595.58.03（不是旧机的 4090D） |
| Python / 环境 | 3.12.3；uv 创建 `/root/h3-venv`，继承系统包 |
| torch / CUDA | `2.8.0+cu128` / 12.8，GPU 可用；torchvision `0.23.0+cu128`、torchaudio `2.8.0+cu128` |
| ComfyUI / CLI | `0.34.0` / `1.20.0`；CLI tracking 已关闭 |
| 容器 RAM 上限 | **90 GiB**（`memory.max=96636764160`），不是宿主机显示的约 755 GiB |
| 磁盘剩余 | 系统盘约 27.6 GiB；50 GiB 模型盘约 **8.0 GiB**，本次无需扩容 |
| 服务 | 已运行，`http://127.0.0.1:8188`，仅监听回环地址 |

沿用上文四个节点包的固定版本/修订，以及 transformers `5.22.0`、
comfy-kitchen `0.2.31`、aimdo `0.4.15`、Triton `3.4.0`。
ComfyUI 归档由服务器直接从
`https://gh-proxy.com/https://codeload.github.com/Comfy-Org/ComfyUI/tar.gz/refs/tags/v0.34.0`
获取；1063 个源码文件均对照官方文件树校验了 Git blob SHA1。
树 SHA 为 `12d5279438bfefc058a269eae805ceab6047777f`，证据见 `core-source-integrity.json`。

新机 Hugging Face 直连超时、阿里 PyPI 返回 403、清华源大文件较慢，
因此模型全部由服务器直下 ModelScope，Python 依赖改用
`https://mirrors.ustc.edu.cn/pypi/simple` 低并发安装。
已完成的 wheel 保存在 `/root/h3-replace/wheels/`，pip 使用 `--find-links` 复用，
避免超时重试时重新下载；没有重新安装或升级基础 CUDA torch。

六个模型均与上文清单一致，已完成**精确字节数、SHA256、safetensors 头/张量数**检查。
其中主模型确为 `minimax_h3_fl2va_pruned_int8_convrot.safetensors`，
编码器确为 `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`。
模型仍在 `/root/autodl-tmp/h3-replace/models/`，通过目录软链接供 ComfyUI 使用，
没有经过本机中转；完整结果见 `models-verification.json`。

### 启动与预检

服务已经启动，不要重复启动。重启仍须先确认队列为空、核对 `comfy.pid` 对应进程；
仅在服务已停止时执行：

```bash
cd /root/h3-replace
setsid nohup bash start-comfy.sh >log/comfy.log 2>&1 </dev/null &
echo $! >comfy.pid
```

`start-comfy.sh` 使用与旧机相同的 `--reserve-vram 4 --enable-triton-backend`，
以及 `--listen 127.0.0.1 --port 8188 --disable-auto-launch --disable-api-nodes`。
启动日志确认 PyTorch attention、DynamicVRAM、双流异步权重卸载已启用；
这只是按需加载/卸载配置就绪，**尚未通过新机的真实 H3 推理观察分层加载行为**。

以下命令只预检和转换，不生成视频：

```bash
ssh -p 30433 root@connect.nmb2.seetacloud.com
export PATH=/root/h3-venv/bin:/root/.local/bin:/usr/local/cuda/bin:$PATH
cd /root/h3-replace
curl -fsS --max-time 15 http://127.0.0.1:8188/system_stats
python run-h3.py --self-test
python run-h3.py \
  --images fixtures/reference_1_480p.png fixtures/reference_2_480p.png fixtures/reference_3_480p.png \
  --video fixtures/reference_10s_480p.mp4 \
  --prompt "$(cat fixtures/prompt.txt)" --seconds 10 --seed 42 --dry-run
```

后续明确要推理时才去掉 `--dry-run`。输出档位仍是一采 `576×320`、二采 `864×480`；
10 秒目标对齐为 243 帧/24fps（10.125 秒），本次只验证了图配置，没有实际输出。

### 验证与审计证据

- torch/torchaudio/GPU 检查、3090 原生 Flash SDPA 小张量检查、`pip check`、
  `--self-test` 和 `py_compile` 均通过；这些不等同于完整 H3 推理验收。
- API 健康，936 个节点可用；转换后的 38 节点 API 图所需类型均已注册。
  三图、一视频、提示词、两个采样器 `247/295`、两个保存输出 `408/299` 均存在，
  六个模型选择正确，upscaler 的 temporal chunking / force unload 均为 true。
- dry-run ID：`h3_20261031_105040_8cc0cb58`；证据位于
  `/root/h3-replace/jobs/h3_20261031_105040_8cc0cb58/`，没有 `submission.json`。
- 最终检查队列 running/pending 均为 0、history 为 0、输出目录无 MP4；
  cgroup `oom=0, oom_kill=0`。汇总证据：`/root/h3-replace/deployment-ready.json`。
- 公开素材仍为服务器直下 Sintel 第 30–40 秒片段：三张不同的 `864×480` PNG，
  非全黑；视频 `864×480`、24fps、240 帧、精确 10 秒，校验见 `fixtures/verification.json`。
- 输入素材和部署/dry-run 证据另已下载到本地当前目录的
  `tmp-h3-deploy-20261031.WiMhGM/`，供审计；本次未推理，因此没有新输出可下载。

未安装无关模板媒体、LayerStyle、rgthree 和 Sage 编译内核，仍是适配后的 CLI/API 部署，
不是原始 GUI 全插件安装。模板包缺失提示不影响此次服务和图转换。
cu128 下优化 CUDA 后端被禁用、Triton 后端可用；未为消除 cu130 提示升级 torch。
**以上为部署完成时的状态；后续 3090 推理实测见下一节，不能套用旧机 11 分钟/19.47 GiB 的结果。**

## 3090 分阶段耗时与显存实测（2026-10-31）

使用部署时的公开素材、原提示词、seed=42，实际执行一次 `run-h3.py`，没有 `--dry-run`。
单个请求同时输入三张 `864×480` 图片、10 秒参考视频和提示词，完成一采 5 步及二采 6 步。
模型和采样参数未改变；未改原工作流、运行脚本或 ComfyUI 核心源码。

- run ID：`h3_20261031_115658_49778d33`。
- prompt ID：`6277b2d5-c466-4732-a98a-f83e311419a1`，history 为 completed/success。
- 后端总耗时 **1185.733 秒（19 分 45.733 秒）**；客户端含预检、转换、轮询及输出验收
  **1188.785 秒**。本次不是多次重复后的均值或热启动成绩。
- 总显存 NVML 采样峰值 **20.181 GiB**；PyTorch allocated / reserved 节点峰值
  **10.087 / 13.750 GiB**。后两者不能相加，也不能替代总显存：DynamicVRAM/VMM
  等分配不全归 PyTorch allocator 统计。
- 容器 RAM / ComfyUI 进程 RSS 采样峰值 **53.452 / 50.913 GiB**；RAM 上限为 90 GiB，
  cgroup `oom=0, oom_kill=0`。容器 RAM 含文件缓存，不是纯张量内存或最小内存要求。

### 计量口径

临时探针包装 ComfyUI 节点执行及 VAE/Qwen/采样函数，在边界执行 CUDA synchronize，
统计包含 CPU、GPU、按需权重搬运及首次初始化/编译的墙钟耗时。
这是模型未驻留 GPU 的单次运行，操作系统文件缓存未清空；同步计时可能改变原有重叠，
因此不是无探针的稳态吞吐基准。loader 节点耗时不代表所有权重搬运时间，后者也会计入实际使用节点。

NVML 总显存、PyTorch 当前占用、RSS 和 cgroup RAM 目标每 0.25 秒采样，
任务内共 4635 个周期样本，实际平均间隔 0.256 秒、最大间隔 2.180 秒；另计各阶段端点。
**下表显存是阶段内整张卡的绝对占用峰值，包含已驻留/缓存，不是该模块独占或新增用量；
采样可能漏掉短峰。** PyTorch allocated/reserved 另在每个节点重置峰值计数，精确到该分配器范围。

### 阶段统计

| 阶段 | 耗时（秒） | NVML 显存峰值（GiB） | 容器 RAM 峰值（GiB） |
|---|---:|---:|---:|
| 模型读取 / LoRA 挂载 | 3.643 | 16.026 | 48.843 |
| 三图读入 / 缩放 | 0.100 | 0.698 | 44.263 |
| 参考视频读帧 | 2.191 | 0.698 | 44.532 |
| 参考条件编码合计（含下面明细） | 71.643 | 20.058 | 50.055 |
| 一采（5 步，含首次初始化） | 350.140 | 20.181 | 50.908 |
| latent upscale | 5.232 | 11.864 | 51.534 |
| 二采（6 步） | 685.881 | 19.708 | 53.452 |
| 一采视频 VAE 解码（反编码） | 18.825 | 10.737 | 49.525 |
| 一采音频 VAE 解码 | 0.562 | 6.106 | 49.017 |
| 一采 H.264/AAC 编码封装 | 1.474 | 10.487 | 49.620 |
| 二采视频 VAE 解码（反编码） | 41.592 | 6.427 | 52.741 |
| 二采音频 VAE 解码 | 0.321 | 1.614 | 51.611 |
| 二采 H.264/AAC 编码封装 | 2.446 | 5.989 | 52.910 |
| 其他计算 / 配置节点 | 0.012 | 16.026 | 51.534 |

其余引擎调度、缓存和计时等开销共 1.671 秒。实际执行了 34 个节点，逐节点数据见 `nodes.csv`。
视频封装使用 FFmpeg；其显存列主要反映此前驻留/缓存，不能当成视频编码器独占用量。

以下是参考条件编码内部的嵌套明细，**不能与上表重复相加**，Qwen 合计也包含其后三项：

| 编码步骤 | 次数 | 耗时（秒） | NVML 显存峰值（GiB） |
|---|---:|---:|---:|
| 图片 VAE 编码 | 3 | 1.293 | 1.425 |
| 参考视频 VAE 编码 | 1 | 43.842 | 3.706 |
| 提示词 / 多模态 tokenize | 1 | 0.022 | 3.706 |
| Qwen 多模态编码合计 | 1 | 24.371 | 20.058 |
| Qwen 图片视觉分支 | 3 | 1.259 | 5.626 |
| Qwen 视频视觉分支 | 10 | 1.624 | 5.626 |
| Qwen 联合语言 Transformer | 1 | 21.003 | 20.058 |

三张图 VAE 编码分别为 0.853 / 0.206 / 0.234 秒。
参考视频读入 240 帧，H3 对齐后 VAE 编码 226 帧，Qwen 分支按 2fps 取样、组成 10 个双帧视觉块。
`KSAMPLER.sample` 函数范围耗时一采 348.324 秒、二采 685.425 秒，
仍包含其内部初始化、按需搬运及首次编译，不是纯 GPU kernel 时间。

### 输出与审计

两份视频均为 243 帧 / 24fps（视频流 10.125 秒），H.264 + AAC 32kHz 双声道：

| 输出 | 尺寸 | 字节数 |
|---|---|---:|
| `pass1_00001-audio.mp4` | 576×320 | 1166674 |
| `pass2_00001-audio.mp4` | 864×480 | 1877711 |

ffprobe、采样画面和音频 RMS 检查均通过：非黑场、非静帧、非完全静音；未做人工参考一致性评分。
运行日志确认 DynamicVRAM / 双流异步卸载，主模型挂载 208 个 LoRA patches，
upscaler 实际执行 temporal chunking 并卸载回 CPU。

- 服务器输出：`/root/comfy/ComfyUI/output/h3_20261031_115658_49778d33/`。
- 服务器报告及探针源码：`/root/h3-replace/benchmarks/h3_profile_20261031_115423/`。
  `report.md` / `report.json` 为汇总；`phases.csv`、`substeps.csv`、`nodes.csv` 为分阶段/逐节点数据；
  `trace.ndjson` 为原始事件；`output-verification.json` 保存媒体校验及 SHA256。
- 本地输入、全部输出（含中间无声视频和封面）、完整 API/UI 图、history、日志和测量源码均已下载至
  **`tmp-h3-profile-20261031.8D553O/`**；入口为 `report.md`，逐节点数据在 `evidence/nodes.csv`。
- 测试后仅移除本次临时探针并恢复常规服务，`cleanup.json` 和 `source-after-check.log` 记录恢复及源码校验。
  为卸载探针进行过空队列重启，GUI 内存 history 因此清空；完整 history 已归档，输出文件保留。

本次证明 24GB RTX 3090 可完成该三图 + 10 秒参考视频、低分辨率一采 + 480p 二采配置；
不代表最低显存需求，也不能推广到 720p、15/20 秒或热启动稳态耗时。

## 磁盘 / PCIe / GPU 利用率复测（2026-10-31）

**结论：本机、本配置下，没有发现磁盘造成 GPU 持续等待的瓶颈；不建议为解决本次采样耗时优先升级磁盘。**
GPU 忙碌率很高，但这不等于达到了理论 FLOPS 峰值，也不代表分层加载本身零开销。

### 测试范围与方法

- 再次实际执行相同三图、10 秒参考视频、原提示词、seed=42，完成一采 5 步和二采 6 步。
  run ID：`h3_20261031_125525_013d907a`；prompt ID：`4811e637-424e-4409-b2f3-b75c58bd0538`。
- 不修改原工作流、运行脚本、服务参数或模型；**不加 CUDA 同步钩子、不重启服务、不清页缓存**。
  原服务 PID `9112` 保持不变，仅运行进程外采集器。
- NVML 采集 GPU 忙碌率、功耗、PCIe RX/TX 和链路；`/proc` 采集进程物理读取/缺页；
  cgroup 采集 I/O 计数和 PSI 压力，另采集共享 NVMe 的吞吐、延迟及忙碌率。
- 目标间隔 0.25 秒，实际平均 0.2519 秒、最大 0.4278 秒；采集器累计使用 21.150 CPU 秒。
  WebSocket 被动接收节点 / progress 事件划分区间，节点耗时是近似值，不能与上轮 CUDA 同步计时逐毫秒对比。
  “首步后”指收到对应采样器第一个 `progress(value=1)` 后，包含后续采样及该节点收尾。

### 观测结果

| 区间 | 秒 | GPU 平均忙碌率 | 进程物理读盘（GiB） | I/O PSI some 增量（秒） |
|---|---:|---:|---:|---:|
| 全程 | 1182.925 | 97.778% | 41.801971 | 1.102876 |
| 参考条件编码 #108 | 71.943 | 92.543% | 13.712346 | 0.173438 |
| 一采 #247 | 347.852 | 99.049% | 21.467564 | 0.686149 |
| 二采 #295 | 686.356 | 99.845% | 1.802170 | 0.035066 |
| 一采收到首步 progress 后 | 270.141 | 99.771% | 0 | 0.003949 |
| 二采收到首步 progress 后 | 565.108 | 99.906% | 0.024296 | 0.009998 |

两个首步后的区间合计 **835.248 秒**，GPU 平均忙碌率 **99.862%**，
物理读取合计仅 **24.879 MiB**，I/O PSI some 仅 **0.013947 秒**；两个区间的 major faults 均为 0。
没有观察到该区间内低 GPU（<50%）与明显物理读盘（>1 MiB/s）同时出现。
因此，持续采样时的权重按需加载主要是从主机 RAM/缓存向 GPU 搬运，不是每步从磁盘重读整套权重。

本次并不是“全缓存、完全没读盘”：进程 `read_bytes` 和 cgroup NVMe `rbytes` 均记录了
**41.802 GiB** 物理读取，但主要集中在加载 / 阶段初始化。全程 I/O PSI some 占比仅 **0.0932%**，
内存 PSI 压力为 0，未发生 OOM。I/O PSI 是等待压力累计，**不是磁盘读操作总耗时**，不能拿它作吞吐计算的分母。

其他交叉证据：

- 模型盘为 XFS `/dev/nvme1n1`（259:1）；物理盘平均读 await **0.2240 ms**、平均忙碌率 **4.115%**。
  这是共享宿主机盘，且 NVMe 忙碌率不能单独代表容量饱和，故仅作旁证，以进程/cgroup计数和 GPU 活动为主。
- 两段首步后区间的 PCIe 均为 **Gen4×16**；RX 采样均值 **219.429 MiB/s**，
  瞬时采样峰值 **21.178 GiB/s**。存在突发搬运，但 GPU 并未因此持续空转。
  PCIe 为速率采样而非精确累计字节，不能精确量化搬运耗时或排除所有短暂等待。
- 稳态 GPU 平均功耗 **343.784 W**，全程平均 **338.095 W**，最高核心温度 66℃。
- 进程全程平均 CPU 占用约 1.458 核（配额 14 核），CPU PSI some 为 0.6664%。
  `task_delayacct=0`，未用进程 `iowait=0` 作为无 I/O 等待的证据；CPU 累计节流微秒也不能直接当 GPU 空闲墙钟时间。

### 验证与审计

两个新生成视频均通过帧数、尺寸、音轨、非黑场/静帧/静音检查；
原始 JSON、`run-h3.py`、启动脚本 SHA256 未变。外部采集器已自动退出，服务继续正常运行，队列为空。

- 服务器报告/源码/原始数据：`/root/h3-replace/benchmarks/h3_io_20261031_125411/`。
- 服务器任务：`/root/h3-replace/jobs/h3_20261031_125525_013d907a/`；
  输出：`/root/comfy/ComfyUI/output/h3_20261031_125525_013d907a/`。
- 本地完整审计包：**`tmp-h3-io-20261031.fjGxLq/`**，包括实际输入、全部输出和完整工作流/history。
  `report.md` 为入口；`evidence/report.json`、`phases.csv`、`after-first-progress.csv` 为统计，
  `samples.ndjson`、`telemetry.csv`、`events.ndjson` 为原始样本/速率/事件，另保留采集与分析源码及媒体 SHA256。

未做全模型常驻显存、强制冷缓存或内存受压对照；结论不外推到其他服务器、慢盘、RAM 不足或更高分辨率配置。


## 链式人物替换实测（2026-11-01，RTX 4080 SUPER 32 GB）

> 提示词写法的通用规则（三图分工、首帧不确定原则、锚点、链式分段）已抽到 [`skill.md`](skill.md)，本节保留工具用法与实测数据。

目标：把原视频里的人物换成角色参考图目录里的角色，输出与原片等长的成片。
只剩一个固定数据点（三图 + 参考视频、一采 576×320、二采 864×480、10 秒）——
本节的链式流程和参数都建立在这个点上，没有改动模型或采样参数。

### 工具

| 文件 | 作用 |
|---|---|
| `run-h3.py` | 服务端单段生成；新增 `--size WxH`（一采尺寸，32 的倍数），二采由 upscaler 按面积缩放 |
| `chain-h3.py` | 服务端链式驱动：切参考片段、拼三张图、逐段生成、记录段间首帧跳变、断点续跑 |
| `assemble-h3.sh` | 拼接所有段、按原片时长重定时、把原片音轨接回去 |

```bash
# 需要先停 Fibo2、启动 ComfyUI
/root/h3-venv/bin/python -u chain-h3.py --task task1 --variant fibo2 --size 480x320 \
  --out /root/h3-replace/chain/task1-fibo2 --seconds 10 --restart-each --max-jump 40 \
  --deadline-hours 5.5
bash assemble-h3.sh /root/h3-replace/chain/task1-fibo2 \
  /root/h3-replace/chain/input/task1-origin.mp4 final.mp4 1064x720
```

输入固定放在 `/root/h3-replace/chain/input/`：`taskN-origin.mp4`、`taskN-{fibo2,qwen}-first.png`、
`sheet.png`（三视图）、`front.png`（单视图）。

### 三张参考图的角色分工

1. `picture1`：本段第一帧。第 0 段用换过脸的首帧，之后每段用上一段的最后一帧（链式延续）。
2. `sheet.png`：三视图，提供正/侧/背造型。
3. `front.png`：单视图，提供脸部与发型。

三张都先缩放到本段的生成画布（二采尺寸）。人物身份几乎全靠图片，文字描述只用来锁定
「不要换装、不要动姿势、不要遮住关键区域」。

### 关键实测数据

- 单段 10 秒（243 帧）耗时 **11.3 分钟**：一采 5 步约 4.3 分钟，二采 6 步约 6.2 分钟，
  模型初始化约 1 分钟，VAE 解码与封装约 1 分钟。整机 GPU 忙碌率接近 100%。
- 每段重启一次 ComfyUI 约 **15 秒**；连续跑 3 段以上时服务会静默退出（无 traceback、
  cgroup `oom=0`），重启可以规避，因此 `--restart-each` 是默认做法。
- 24 段（240 秒）拼接 + 重定时 + 重新编码 720p 约 **1 分 40 秒**（纯 CPU）。
- 画布与尺寸：`--size 480x320` 时一采 480×320、二采 736×480；竖屏 `--size 320x448`
  时二采 480×672。upscaler 会按 32 像素对齐自行决定二采宽度，所以脚本只校验宽高比，
  不强制精确像素。

### 三个必须踩过的坑

1. **每段多 0.125 秒会累积漂移**。10 秒生成 243 帧 = 10.125 秒，24 段就是 243 秒，
   比原片多 3 秒。`assemble-h3.sh` 用 `setpts` 把拼接结果重定时回原片时长，再接原音轨；
   参考片段也按 243 帧切，避免段与段错位。
2. **相邻段首帧不可能像素级一致**。灰度 96×64 的平均绝对差在 18–40 之间波动，
   换 seed 几乎不改变这个值（同段 seed 42/43 分别是 27.31/27.30）——它反映的是光照和
   微小构图变化，不是身份或场景断裂。阈值设 40；超过就换 seed 重跑，实际 24 段里只有
   末段触发过一次。
3. **服务重启要等锁释放**。`start-comfy.sh` 用 flock 互斥，杀掉旧进程后立刻启动会报
   `Another inference service is running`。驱动里先轮询 `flock -n` 再启动。

### 质量观察（task1，240 秒，24 段）

- 身份：整片稳定，银白发/侧编发/铃铛蝴蝶结/蓝眼贯穿全程，没有回退成原角色。
- 构图与动作：跟随原片镜头、体位和时间点；段间没有明显跳变。
- 素材泄漏：中后段（约 100 秒后）服装细节会向参考图靠拢（袖口十字徽章、红色滚边），
  提示词已明确禁止，仍会局部发生；要彻底解决得把参考图裁到只剩头部。
- 后段背景出现噪点、结尾几秒发糊：长链的累积误差，缩短段数或每段重建参考可缓解。
- 首帧的衣物会把局部相接区域盖住一版，但视频动作会随原片推进逐步露出，不必单独重跑段。

### 未完成部分

240/238/142 秒三条原片 × 两个首帧变体 = 124 段，按 11.3 分钟/段约 23 小时。
2026-11-01 夜间完成 `task1-fibo2`（24 段）并开始 `task3-fibo2`（竖屏，14 段中取前几段验证）。
`chain-h3.py` 可按 `--start/--count` 续跑，`state.json` 记录已完成段，不要删除输出目录。
