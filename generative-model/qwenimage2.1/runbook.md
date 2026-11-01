# Qwen-Image-2.1-Turbo：本地多图编辑与文生图

## 当前状态

**已完成部署与真实 GPU 运行验收，服务保持运行：`127.0.0.1:8191`。** 已从本机通过 `run.py` 完成四次 SSH 上传/生成/下载，包括四图 2048×2048、单图 2048×2048 和文生图。运行通过不代表严格语义/画质合格，见文末观察与限制。

- 服务器：`ssh -p 30433 root@connect.nmb2.seetacloud.com`，RTX 3090 24 GiB，容器 RAM 上限 90 GiB。
- 模型是官方 **Qwen/Qwen-Image-2.1-Turbo 完整 BF16**，不是旧 Qwen-Image/Edit、LoRA 或量化替代品。
- 模型、素材由服务器直接下载，不经本机中转；推理不调用任何外部图片处理或提示词服务。
- 为腾盘，经用户授权停止 Fibo2 并删除其四个权重，释放 **23.718 GiB**；旧环境、配置、脚本、输入、输出、日志和其余 H3 模型保留。Fibo2 目前不可直接运行，H3 仍缺先前清理的主模型/编码器。
- **Qwen Research License 仅限非商业研究/评估，商用需另行授权**：[官方许可](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/raw/main/LICENSE)。不是 Apache/MIT 许可。

## 1. 本机直接运行（推荐）

本机需要 Python 标准库、系统 SSH/SCP、已有的 SSH 密钥/agent 和可信主机指纹。无需本机 CUDA/Torch，无需手动建隧道；同目录的两个 Python 文件需一起保留。脚本开头均有用法说明。

```bash
cd /Users/yingtiankai/Desktop/Runbooks/video-generation/qwenimage2.1

# 四图编辑；按路径顺序编号，含空格的文件名加引号
uv run --no-project python run.py \
  --images /path/one.png /path/two.png /path/three.png /path/four.png \
  --prompt '将 <image1>、<image2>、<image3>、<image4> 按顺序组合成 2×2 分镜。保持人物身份、服装、场景细节和原有色彩，不添加文字。' \
  --width 1024 --height 1024 --seed 42 --output result.png

# 单图编辑
uv run --no-project python run.py \
  --images /path/one.png \
  --prompt '将 <image1> 变为细腻的水彩插画，保持主体、表情和构图，不添加文字。' \
  --output watercolor.png

# 文生图：不传 --images
uv run --no-project python run.py \
  --prompt '一只坐在窗边的橘猫，柔和的秋日阳光，细腻真实的毛发，专业摄影，背景自然虚化。' \
  --width 2048 --height 2048 --output cat.png
```

默认 1024×1024、seed 42；可传 0–4 图。输出 PNG 和同名 JSON（job ID、服务端耗时、SHA256、服务器原件路径），拒绝覆盖已有输出。生成图片保留模型原生 RGBA 通道；参考图透明度也保留。

1024 是本服务的速度/资源折中，不是模型原生示例的 2048 档位；需要更多细节时可显式指定 `--width 2048 --height 2048`，四图 2048 也已真实跑通，但细节/布局仍需人工审计。

**Qwen 原生标签是 `<image1>`，不是 Fibo2 的 `<image_1>`。** 为兼容旧调用，服务会将后者明确转换为前者，并在任务记录中保存原始 `prompt` 与 `effective_prompt`。标签必须引用实际上传的图；无图任务不能引用图片标签。不使用 Gemini 或其他提示词扩写。

流程是 SCP 上传媒体到唯一中转目录 → SSH 调用服务器客户端 → 服务器回环 HTTP 调用已加载模型 → SCP 下载 PNG/元数据 → 签名和 SHA256 校验 → 本机独占创建输出文件。成功后只清理本次中转目录；原始任务、输入和输出仍保留在服务器 `jobs/`。失败/中断保留中转文件，路径会先打印；推理可能仍在执行，先检查远端输出，**不要盲目重跑**。脚本不自动下载模型、启动服务或重试推理。

SSH/SCP 加密并保护传输完整性，沿用系统主机校验，不关闭校验、不写入密码。**落盘文件仍为明文**；不要把回环服务直接暴露公网。

## 2. 服务器客户端与接口

服务器本机调用：

```bash
cd /root/qwenimage2.1
/root/fibo2-venv/bin/python run-qwenimage.py \
  --images fixtures/image_1.png fixtures/image_2.png fixtures/image_3.png fixtures/image_4.png \
  --prompt 'Create a 2 by 2 storyboard using <image1>, <image2>, <image3>, <image4> in order. Preserve the characters and colors. No text.' \
  --width 1024 --height 1024 --output my-edit.png
```

`GET http://127.0.0.1:8191/health` 返回 ready/busy、模型名、GPU。`POST /generate` 使用 `Content-Type: application/json`，图像字段在顶层，值为纯 base64（无 data-URL 前缀）；无图时省略全部图像字段。

```json
{
  "image_1": "BASE64_OF_IMAGE_1",
  "image_2": "BASE64_OF_IMAGE_2",
  "prompt": "Combine <image1> and <image2> into a natural photograph.",
  "width": 1024,
  "height": 1024,
  "seed": 42
}
```

- 0–4 图、连续编号；单帧 PNG/JPEG/WebP，原文件每张 ≤10 MiB、≤1600 万像素。EXIF 方向正常化，转 RGBA，实际参考图缩放遵循官方 pipeline，不强制裁成方图。
- 整个 body ≤64 MiB；提示词非空，UTF-8 ≤16384 bytes；宽高为 512–2048 的 32 倍数。seed 是 `[0, 2**63)` 整数。
- 成功直接返回 `image/png`，头有 `X-QwenImage-Job-ID`、`X-Elapsed-Seconds`。非法输入 400，错误 Content-Type 415，单 GPU 忙碌时 503，不排队。
- 固定官方 Turbo 8 步、`true_cfg_scale=1.0`、`use_kv_cache=True`，不暴露 steps/CFG/自定义 sigmas，避免误用 40 步默认值或错误采样表。
- 不提供 mask、训练、多图 batch 或 >4 图接口；不是带鉴权、隔离、队列的公网多租户服务。

## 3. 模型、版本及显存策略

官方模型页：[Qwen/Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo)。服务器 HF 直连不可达、hf-mirror 403，使用官方 ModelScope 同名库固定快照：

```text
4954d69c119759f085aa36bd5e66e27647cdee60
https://modelscope.cn/models/Qwen/Qwen-Image-2.1-Turbo/resolve/4954d69c119759f085aa36bd5e66e27647cdee60/
```

必要文件 17 项，合计 **32,451,622,214 bytes（30.223 GiB）**。下载后全部按大小和 SHA256 校验，再检查四个 safetensors 的头/张量和 transformer 分片索引；通过后才创建 `models.complete`。四个权重 SHA256 也已对照 HF 官方文件页，均与 ModelScope 清单一致：

| 相对路径 | bytes | SHA256 |
|---|---:|---|
| `text_encoder/model.safetensors` | 17534340584 | `a1eee8233f3063466dd3b06e857b0a2cf13a77c543908374771d51a546beb2e0` |
| `transformer/diffusion_pytorch_model-00001-of-00002.safetensors` | 9968332504 | `6cccd922767f01694461bdcf1f34ea7b771f3442e4d5e338ea9aa55277431cc0` |
| `transformer/diffusion_pytorch_model-00002-of-00002.safetensors` | 4261951904 | `69f53ebb063d2f606bdaff7e22e0e2144f28d978d73dc75e6e4293fc5982eab5` |
| `vae/diffusion_pytorch_model.safetensors` | 675508656 | `71879ffd5321e6d10c3c87513e2b474b1252efa7f3dec2969214a9bf06a6dd5c` |

| 运行环境 | 已验证值 |
|---|---|
| GPU / 驱动 | RTX 3090，24576 MiB / 595.58.03 |
| Python | 3.12.3，复用 `/root/fibo2-venv`，**不要删除该环境** |
| torch / CUDA | 2.8.0+cu128 / 12.8，CUDA 可用 |
| diffusers / transformers | 0.42.0 / 5.22.0 |
| accelerate | 1.15.0 |
| 容器 RAM 限额 | 90 GiB，不能按宿主机 `free -h` 推断可用内存 |

已安装 Diffusers 0.42.0 原生包含 `QwenImage21Pipeline` 和模型配置采样表支持；不需要再从源码升级或重装 CUDA torch，旧环境未被修改。模型初始化核对 BF16 组件和固定 `sample_sigmas`：

```text
[1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]
```

这是 checkpoint 保存的 8 步表，不手动用线性 sigma 替代。官方说明单设 `num_inference_steps` 不会覆盖保存的采样表；实际回调必须恰好 8 次。

7B 是视觉 DiT 的规模，不包含单独的 Qwen3-VL 编码器；仅编码器文件就约 17.53 GB，因此不能按“7B×2 bytes”估算整套模型磁盘/显存。

显存策略：BF16、编码器/VAE 组件级卸载、Transformer 官方 **4 层一组 CPU/GPU group offload**、逐 token 前馈分块（1024 tokens）、VAE tiling、官方 prefix KV cache、`expandable_segments:True` 分配器、参考图 `output_resolution=1024`、CPU 随机数 generator（官方用法）。

**这里确实使用了按层分组加载**：DiT 每个采样步都会按组在 CPU RAM/GPU 之间搬移，不是全模型常驻显卡；未启用磁盘 offload 或量化，也未降低参考图分辨率、修改采样表。为避免组件卸载和 group offload 冲突，只给编码器/VAE 添加 Accelerate 组件 hook，在 DiT 前显式释放它们；DiT 使用 Diffusers 自带 group offload。依赖 CPU RAM/PCIe，会比整块模型常驻 GPU 慢；24 GiB 验收不能推导为更小显存也可用，超大/极端比例输入可能有不同峰值。

前馈分块只针对固定版本的 QwenImage21 token-wise SwiGLU，数学上等价（不同 GEMM 形状可有舍入差异）；CPU FP32/CUDA BF16 小模型对照分别测得最大绝对误差 `2.98e-8` / `0`，见 `ffn-equivalence.json`。不要在未重新验证架构时迁移这段 patch。

## 4. 服务器文件与维护

```text
/root/qwenimage2.1/
  run-qwenimage.py
  start-qwenimage.sh
  service.pid
  download-models.sh
  verify-models.py
  models.aria2
  models.complete
  metadata/model-manifest.json
  models-verification.json
  runtime-verification.json
  previous-model-cleanup.json
  fixtures/
  jobs/<job_id>/             # 原始图 .source、正常化 PNG、request/result JSON、output.png
  log/
/root/autodl-tmp/qwenimage2.1/models/Qwen-Image-2.1-Turbo/
/root/fibo2-venv/            # 复用已验证环境，仅清理旧模型，不删除环境
```

需要补下/重新核验模型时，**先停服**，再在服务器执行 `bash /root/qwenimage2.1/download-models.sh`，不要在推理时覆盖已映射的权重，也不要手工创建完成标记。脚本会先移除旧完成标记，再断点续传/逐项校验。当前 `models.aria2` 使用稳定的官方 resolve 地址，临时 CDN 加速地址另存为历史快照，避免依赖过期签名。新机器仍需先小文件/目标 wheel 连通性和速度预检，再选择源；不要假设 download.pytorch.org 可用。本次没有安装或升级 Python 依赖。

新机若没有这个环境，先单独安装/验证兼容 CUDA 的 torch 2.8.0、torchvision 0.23.0，再用 uv 创建继承基础 CUDA 包的环境；本次无需执行以下重建命令：

```bash
uv venv --system-site-packages --seed --python /root/miniconda3/bin/python /root/fibo2-venv
# 新机先对镜像小文件预检；无有效下载进展超过 10 分钟就换源，不盲等。
timeout 600 /root/fibo2-venv/bin/python -m pip install \
  --index-url https://mirrors.aliyun.com/pypi/simple/ --retries 30 --timeout 60 \
  -c /root/qwenimage2.1/constraints.txt diffusers==0.42.0 transformers==5.22.0 \
  accelerate==1.15.0 safetensors==0.8.0 pillow==13.0.0 sentencepiece==0.2.2
/root/fibo2-venv/bin/python -m pip check
/root/fibo2-venv/bin/python -c 'import torch; print(torch.__version__,torch.version.cuda,torch.cuda.is_available(),torch.cuda.get_device_name(0))'
```

约束文件和依赖冻结记录随审计包保存；新机复制这些小型脚本/配置后，仍应在服务器直接下载依赖和权重，不能把本机当下载中转。

查看状态：

```bash
curl -fsS http://127.0.0.1:8191/health
ps -p "$(cat /root/qwenimage2.1/service.pid)" -o pid,args
tail -n 30 /root/qwenimage2.1/log/service.log
```

未注册开机自启。确认没有旧进程、8191 未占用后，重启服务器可手动启动：

```bash
cd /root/qwenimage2.1
setsid nohup bash start-qwenimage.sh >"log/service-$(date +%Y%m%d-%H%M%S).log" 2>&1 </dev/null &
echo $! >service.pid
# 等待 READY，再检查 /health；不要重复启动。
```

停止前先确认 `busy=false`，核对 `service.pid` 对应 `run-qwenimage.py --serve --port 8191`，再只终止该 PID；不要全局 pkill Python。运行设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`HF_HUB_DISABLE_TELEMETRY=1`，并以 `local_files_only=True` 读取所有组件，无 `trust_remote_code`。

## 5. 验收与审计

### 最终配置的运行验收

| 任务 | 输出 | 服务端耗时 | NVML 采样显存峰值 | Torch allocated / reserved 峰值 |
|---|---|---:|---:|---:|
| 四张 Sintel 参考图，兼容 `<image_1>` 提示词 | 1024×1024 RGBA PNG | 118.176 s | 21634 MiB（21.13 GiB） | 19.798 / 20.807 GiB |
| 官方游艇线稿＋中文写实编辑指令 | 2048×2048 RGBA PNG | 165.216 s | 18278 MiB（17.85 GiB） | 17.240 / 17.527 GiB |
| 无参考图，中文秋日海报指令 | 2048×2048 RGBA PNG | 156.588 s | 17234 MiB（16.83 GiB） | 16.421 / 16.508 GiB |
| 四张 Sintel 图，原生标签＋明确分格指令 | 2048×2048 RGBA PNG | 192.999 s | 20716 MiB（20.23 GiB） | 19.818 / 19.908 GiB |

四次均是从**本机 `run.py`** 经真实 SSH/SCP 上传/取回，恰好 8 步，PNG 尺寸/通道/文件哈希与服务器原件一致；四图输入 SHA 均不同。0、1、4 图均实测，2、3 图只做了入参自检。六种非法输入返回 400、错误类型 415、最终配置忙碌请求返回 503；包装层 4 项测试及 CPU/CUDA 前馈等价性检查通过，语法与两个入口 `--help` 通过。

耗时从输入准备并落盘后开始，包含模型调用和输出编码，不含首次加载模型、SSH 上传/下载。NVML 每 0.5 秒采样，不是瞬时硬峰；只统计各成功任务的时间窗，不把前期 OOM 的峰值混入。四次 GPU 采样平均利用率约 30.2%、42.9%、40.7%、44.9%，最大均为 100%：按层分组搬移带来 CPU/PCIe 等待，不能声称显卡全程满载，也不能据此认定为磁盘 offload。数字是单次验收数据，不是吞吐承诺。

### 调校中的两次失败没有隐藏

- `qwenimage_20261031_163343_7984a39e`：单纯组件 CPU offload＋VAE tiling，四图 1024 在 FFN 中 CUDA OOM。
- `qwenimage_20261031_163948_8275b0d2`：加 FFN=1024 和 expandable segments 后，仍在 attention/rotary 中 CUDA OOM。
- 因此最终采用 **4 层 group offload**，保留完整 BF16、参考图 1024 档和 KV cache；最终四次没有运行失败，容器 `oom/oom_kill=0`。这里的容器 OOM 指 CPU 内存，不等于前两次 CUDA OOM 不曾发生。

失败请求、stderr、`error.json` 和两个旧服务日志均保留在审计包。不要在这张 24 GiB GPU 上把配置改回单纯 `enable_model_cpu_offload()` 后假定四图仍能运行。

### 画质与指令遵循观察

单图能把游艇线稿变为海上船只图，中文海报主标题“秋日来信”可读；但**并非全部指令严格满足**：1024 四格例出现参考对应偏差，2048 四图例出现额外分格/细节变化，海报仍多出未要求的小字/乱码。个别图片还有可见细节/条纹瑕疵；本次没有另做不分块 VAE 解码的对照。所有原图保留供比较，不将“成功返回 PNG”包装成严格的画质验收通过，也不承诺一定优于 Fibo2。

输入、输出、请求/结果、清理记录、下载及资源日志保存在本机临时审计目录 `tmp-qwenimage-acceptance-20261031.bETDUi/`，不含模型权重；目录以 `.gitignore` 排除大文件，但保留供审计。

本地/服务器结构自检：`python run-qwenimage.py --self-test`。它只检查入参约束，不能替代上面记录的真实 GPU、传输和画质观察。

本次真实验收证据包括：`evidence/deployment-verification.json`、`evidence/jobs/`、`evidence/gpu-acceptance.csv`、两个失败 stderr/服务日志、`evidence/test-wrapper.py`、`evidence/test-ffn.py`。公共输入来源分别为服务器原已下载的 Sintel 片段和官方 ModelScope 快照的 `assets/turbo-boat-input.png`，来源/哈希在 `inputs/` 中。审计目录 `SHA256SUMS` 可用 `shasum -a 256 -c SHA256SUMS` 验证。

## 6. 角色替换（三视图参考 → 场景换人）实测配方

用户两条原命令（`qwenimage_20261031_192010_6e6a0dd6`、`qwenimage_20261031_192431_077ac406`）基本原样返回场景图，只出现轻微改色。核对服务端 `HuggingFace diffusers 0.42.0 QwenImage21Pipeline` 源码与官方 Qwen-Image-2.1 editing system prompt 后，原因不是 seed/步数/分辨率，而是提示词写法：

1. **多图必须用 `<image1>`/`<image2>` 标签**（服务会把 `<image_1>` 转成 `<image1>`）。自然语言「图一/图二」不会把文字绑定到具体图像，模型就按「编辑第一张图、其余当弱参考」处理。官方 rewriter 明确禁止「图1、第一张图、image A」。
2. **要显式写出每张图的职责与保留项**：哪张是画布（保留背景/构图/机位/光线/姿态）、哪张只提供身份与服装。只写「把人物换成图二的角色」信息量不足。
3. **身份来自参考图时「指图」，不要重新描述人物**。官方原话：用文字重述长相会让模型重画一个符合描述的人，反而降低一致性。可以补充的只有「必须消失的原图元素」（如原发色、原发饰、原水手服裙摆）。
4. **画布图放第一位**，角色图第二位；`--width/--height` 按画布图比例传。顺序反过来（角色图第一）也会跑，但标签必须跟着改。
5. **参考图不必裁成单视角**：1024×1024 的四视图整图可直接用，只需在提示词写明「只取其中一个视角，不复制四视图排版和站姿」。
6. **不要把脸/胸像特写当第三张参考**：`qwenimage_20261031_194712_bb74758f` 直接把特写当画布输出（整张脸），构图全丢。要增强脸就换更清晰的角色立绘，别加特写。
7. 只写「整体替换」会留下原服装/原发饰（`..._183227_f77a9280` 留下黑发，`..._194050_0f27160f` 留下绿色发饰和侧马尾）；必须在同一段里点名要换掉的部位。

可直接复用的提示词骨架：

```text
以 <image1> 为画布：保持 <image1> 的<背景、构图、机位、光线、人物姿态>完全不变。
将 <image1> 中<对象>的身份和服装整体替换为 <image2> 中的角色：脸部五官、发型、发色、发饰和整套服装全部沿用该角色的外观，<必要的穿着结果>。
<原图中必须消失的具体元素：原发色、原发饰、原服装的部位>都换成 <image2> 里的这套服装。
<image2> 只用于角色身份与服装参考，只取其中一个视角的角色设定，不复制它的白色背景、四视图排版和站姿。
保持原有动漫插画风格，画面中人物数量不变，不添加任何文字。
```

通用的多图编辑提示词规则另见同目录 `skill.md`（与具体案例无关、可直接复用；作为 standalone skill 文件读取）。

## 风险与边界

研究许可限制、单 GPU 串行、无公网鉴权、输入输出持续明文留盘；不自动清理用户任务。未承诺长期稳定性、主观效果必然优于 Fibo2、所有分辨率/素材组合或真实断网恢复；没有调用额外提示词模型。用完整描述明确主体、场景、参考图对应关系，最终效果请审计实际输出。
