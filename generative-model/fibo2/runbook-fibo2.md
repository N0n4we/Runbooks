# Fibo 2.0 Edit Turbo：本地多图编辑服务

> **当前状态（2026-10-31，迁移新机后）**：新服务器 `ssh -p 14499 root@connect.weste.seetacloud.com`（RTX 4080 SUPER，可见显存 32760 MiB，驱动 595.71.05；数据盘 150G，部署后余约 99G，容器内存 503 GiB），Fibo2 服务已运行在 `http://127.0.0.1:8190`，模型为 `briaai/Fibo-2.0-edit-turbo` 完整 BF16。本机 `run.py` 的 `HOST`/`PORT` 已指向新机；`qwenimage2.1/run.py` 仍指向已删除的旧机，不要再跑。
>
> 旧机（`connect.nmb2.seetacloud.com:30433`，RTX 3090）已由用户删除。旧机上 Fibo2 曾为给 Qwen 腾盘清理过权重，那台机器的历史验收记录保留在下方，不代表新机状态。
>
> 迁移后已在新 GPU 上真实跑通 8 次编辑，其中 3 张角色替换成片见 §8；1024×688 的 Transformer reserved 峰值约 20.1 GB。

## 历史验收状态（2026-10-31，旧机清理前）

- 服务器：`ssh -p 30433 root@connect.nmb2.seetacloud.com`（旧机，已删除）。
- 服务已运行：`http://127.0.0.1:8190`，支持 **1–4 张参考图＋提示词 → 一张 PNG**。
- 已真实验收四图 1024×1024、单图英文/中文 768×768；不是仅完成下载或健康检查。
- **图片只上传到这台自己的服务器**；推理完全离线，不使用 Gemini、Google API 或其他外部图片处理服务。
- 为腾盘，经用户授权停止 H3、删除其主模型和 32B 编码器，释放 **34.14 GiB**；其他 H3 模型、环境、工作流、输入、成片和测试证据保留。H3 目前不可直接运行。
- Fibo2 部署后数据盘约剩 **19 GiB**，系统盘约剩 **27 GiB**，无需继续删除 H3 文件。

## 1. 直接使用

### 推荐：本机 `run.py` 自动 SSH 上传、生成、下载

不需要手动建隧道，只需本机 Python 标准库和系统 `ssh`/`scp`；沿用现有 SSH 密钥/agent 和主机校验，不处理密码或关闭主机校验。默认连接当前服务器，迁移时修改 `run.py` 顶部的 `HOST`、`PORT`、`REMOTE`。

```bash
cd /Users/yingtiankai/Desktop/Runbooks/video-generation/fibo2
uv run --no-project python run.py \
  --images /path/one.png /path/two.png /path/three.png /path/four.png \
  --prompt 'Create a 2 by 2 storyboard: top-left from <image_1>, top-right from <image_2>, bottom-left from <image_3>, bottom-right from <image_4>. Preserve the characters and colors. No text.' \
  --width 1024 --height 1024 --seed 42 --output result.png
```

支持 1–4 图、中文提示词和含空格的文件名（shell 中给文件名加引号）。`run.py` 复用同目录 `run-fibo2.py` 的输入校验，因此两个文件要一起保留。

流程：SCP 上传到服务器唯一临时目录 → SSH 调用已部署的 `run-fibo2.py` 客户端（复用已加载服务，不另加载模型）→ SCP 下载 PNG 和元数据 → 校验 PNG 签名/SHA256 → 拒绝覆盖地写入本机 `result.png`、`result.json`。元数据包含远程 job ID、服务端耗时、SHA256 和服务器保留的输出路径。

成功后只清理本次上传中转目录，服务器 `jobs/<job_id>/` 中的输入输出仍保留。失败/中断保留远程中转文件，路径会提前打印；断线后推理可能仍在继续，**不要直接重跑导致重复生成**，先检查该目录的 `output.png`/`output.json`，必要时用 SCP 取回。脚本不自动启动服务或重试推理，服务必须已经运行。

包装脚本本地测试：`uv run --no-project python -m unittest -v test_run`。

新机（2026-10-31）上遇到过两次生成成功、但 `run.py` 最后一步 SCP 报 `Connection closed`（SSH 长连接被中途掐断）。服务端结果仍在 `/root/fibo2/uploads/<本次目录>/output.png`，手动取回即可，**不要重跑**：

```bash
ssh -p 14499 root@connect.weste.seetacloud.com 'ls -1t /root/fibo2/uploads | head -3'
scp -P 14499 root@connect.weste.seetacloud.com:/root/fibo2/uploads/<dir>/output.png ./x.png
scp -P 14499 root@connect.weste.seetacloud.com:/root/fibo2/uploads/<dir>/output.json ./x.json
ssh -p 14499 root@connect.weste.seetacloud.com 'rm -rf -- /root/fibo2/uploads/<dir>'
```

### 原始方式：手动 SSH 隧道＋HTTP 客户端

本机终端一建立 SSH 隧道，并保持此终端打开：

```bash
ssh -N -L 18190:127.0.0.1:8190 -p 14499 root@connect.weste.seetacloud.com
```

本机终端二运行客户端；按 `--images` 的文件顺序对应 `<image_1>`…`<image_4>`：

```bash
cd /Users/yingtiankai/Desktop/Runbooks/video-generation/fibo2
curl -fsS http://127.0.0.1:18190/health
uv run --no-project python run-fibo2.py \
  --url http://127.0.0.1:18190 \
  --images /path/one.png /path/two.png /path/three.png /path/four.png \
  --prompt 'Create a 2 by 2 storyboard: top-left from <image_1>, top-right from <image_2>, bottom-left from <image_3>, bottom-right from <image_4>. Preserve the characters and colors. No text.' \
  --width 1024 --height 1024 --seed 42 --output result.png
```

返回 `result.png` 和 `result.json`（job ID、服务端耗时、输出 SHA256）。客户端只需 Python 标准库，本机无需安装 Torch 或下载模型；已有同名输出时拒绝覆盖，不自动重试推理。

单图只传一个文件，提示词可以用中文，例如 `--prompt '将 <image_1> 改为黑白铅笔素描，保持原构图，不要文字。'`。两张/三张图同理，标签从 1 连续编号。**多参考图的“换人不换构图”不要用这种自然语言提示，必须按 §8 的 VGL JSON 写**：模型卡声明该 checkpoint 只用英文训练，自然语言多图指令会被当成“重新构图”。

服务器内也可直接使用，无需隧道：

```bash
cd /root/fibo2
/root/fibo2-venv/bin/python run-fibo2.py \
  --images fixtures/image_1.png fixtures/image_2.png fixtures/image_3.png fixtures/image_4.png \
  --prompt "$(cat fixtures/prompt-four.txt)" \
  --width 1024 --height 1024 --output my-edit.png
```

## 2. 接口与限制

`GET /health` 返回 `ready`、`busy`、模型名和 GPU。`POST /edit` 使用 `Content-Type: application/json`，图像字段在 JSON **顶层**，值为纯 base64，不带 data-URL 前缀：

```json
{
  "image_1": "BASE64_OF_IMAGE_1",
  "image_2": "BASE64_OF_IMAGE_2",
  "image_3": "BASE64_OF_IMAGE_3",
  "image_4": "BASE64_OF_IMAGE_4",
  "prompt": "Combine <image_1>, <image_2>, <image_3> and <image_4> into a storyboard.",
  "width": 1024,
  "height": 1024,
  "seed": 42
}
```

- 必须有 1–4 图、编号连续；不能引用未上传的图片标签，不接受未知请求字段。
- 单帧 PNG/JPEG/WebP，每个原文件 ≤10 MiB、≤1600 万像素；整个 HTTP body ≤64 MiB。EXIF 方向正常化后转 RGB。
- 提示词非空，UTF-8 编码 ≤4096 bytes。宽高均为 256–1024 的 16 倍数，默认 768×768；seed 默认 42，范围 `[0, 2**63)`。
- **固定 4 步、guidance_scale=1.0、BF16**，符合 Edit Turbo 用法；不提供无图文生图、mask 或训练接口。
- 成功直接返回 `image/png`，响应头有 `X-Fibo2-Job-ID`、`X-Elapsed-Seconds`。
- 非法输入 400，非 JSON 类型 415；同一时间只执行一个 GPU 请求，忙碌时返回 503，不排队。
- 服务器的 `jobs/<job_id>/` 保存原始上传文件、正常化输入、`request.json`、`result.json`、`output.png`；失败保存 `error.json`。会持续占盘，图片有隐私时需自行制定清理策略。

## 3. 本地提示词路径与显存策略

使用官方 `diffusers==0.42.0` 的 `BriaFibo2Pipeline`。编辑模式原生接受自然语言指令，`_get_edit_prompt(prompt, num_images)` 在本机包装为包含图像占位符和 `edit_instruction` 的 JSON；四张图是同一任务的四个参考，不是四个独立 batch。任务 JSON 保存 `encoded_edit_prompt` 作为审计证据。

**没有调用官方示例的 Gemini/bria-vgl 自动扩写，也没有另外安装 caption/VLM 模型。** 这是原生 instruction-only 编辑路径，不是自动补全完整 VGL 描述，不保证与外部 VGL 扩写相同的质量。如果已有完整 VGL，可将带 `edit_instruction` 的 JSON 字符串作为提示词传入。

模型、processor、字体均从本地路径读取，`local_files_only=True`；运行设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`HF_HUB_DISABLE_TELEMETRY=1`，不启用 `trust_remote_code`。

显存优化为 `enable_model_cpu_offload()` 和 VAE tiling：按**组件**在 CPU/GPU 间搬移，依赖 CPU 内存及 PCIe；**不是逐层 sequential offload，也没有启用 disk offload 或量化**。CPU 内存中的权重不等于每步重新从磁盘读模型。24 GiB GPU 已通过本次四图验收，但不据此承诺更低显存、更高分辨率或所有输入组合。

## 4. 环境与服务器目录

| 项目 | 已验证值 |
|---|---|
| GPU / 驱动 | RTX 4080 SUPER，32760 MiB / 595.71.05（迁移后新机；旧机为 RTX 3090 24576 MiB / 595.58.03） |
| Python | 3.12.3；独立环境 `/root/fibo2-venv`，继承基础 CUDA torch |
| torch / torchvision | 2.8.0+cu128 / 0.23.0+cu128；CUDA 12.8 可用 |
| diffusers / transformers | 0.42.0 / 5.22.0 |
| accelerate / safetensors | 1.15.0 / 0.8.0 |
| Pillow / huggingface-hub | 13.0.0 / 2.4.0 |
| 容器 RAM 限额 | 新机 `free -g` 总内存 503 GiB；旧机为 90 GiB 限额 |

```text
/root/fibo2/
  run-fibo2.py                 # 服务端与标准库客户端同一脚本
  runbook-fibo2.md
  start-fibo2.sh               # 离线环境、固定解释器和启动目录
  service.pid
  constraints.txt
  install-runtime.sh          # 本次安装记录，复用服务器 wheels 缓存
  download-models.sh          # 断点下载，全部 SHA256 通过后写 models.complete
  models.aria2
  metadata/model-manifest.json
  fixtures/                   # 4 张公开输入、提示词、来源
  jobs/                       # 每次推理输入输出及审计记录
  acceptance/                 # 接口验收证据
  log/
  deployment-verification.json
/root/fibo2-venv/
/root/autodl-tmp/fibo2/models/Fibo-2.0-edit-turbo/
```

依赖由服务器直接下载安装，先验证继承的 torch/CUDA，再安装其余依赖，没有重装 H3 的环境。USTC 源部分 wheel 慢，改为服务器缓存完整 wheel 后安装，后续 `pip check`、5335 个带 SHA256 的包文件 RECORD 校验和真实推理全部通过。确切版本见 `constraints.txt`、`log/python-freeze.txt`、`package-integrity.json`。

新机器不要假设官方 PyTorch 下载源可达；先对小文件/目标 wheel 预检，失败就换 PyPI 镜像。阿里源 pip 使用 `--index-url https://mirrors.aliyun.com/pypi/simple/ --retries 30 --timeout 60`；uv 不接受这两个 pip 重试参数，应在 shell 加超时重试。单次超过 10 分钟无有效下载进展需终止换源，不要重新盲装 CUDA 包。当前机器不需要重装环境。

## 5. 模型下载与完整性

模型：`briaai/Fibo-2.0-edit-turbo`，完整 BF16 权重，无第三方量化。HF 直连不可达、hf-mirror 返回 403，本次由**服务器直下 ModelScope 同名公开仓库**，固定快照：

```text
086fe766569224bc8f8c4798f98f3df91b2020b4
https://modelscope.cn/models/briaai/Fibo-2.0-edit-turbo/resolve/086fe766569224bc8f8c4798f98f3df91b2020b4/
```

必要文件 21 项合计 **25,479,015,764 bytes（23.729 GiB）**，大小与 SHA256 全部按该镜像清单核验；四个 safetensors 的头部、张量数和 transformer 分片索引也已校验。四个权重如下（其余配置、processor、字体见 manifest）：

| 相对路径 | bytes | SHA256 |
|---|---:|---|
| `text_encoder/model.safetensors` | 8875719408 | `f0b24fc5c8d7c0ea6ca909da14c20af5cac53f22506be942c6c956f838b9630f` |
| `transformer/diffusion_pytorch_model-00001-of-00002.safetensors` | 9991176550 | `bb3bcab301489d31bbe5226a53b5f7765f420616860e4abf7b1116a66a705090` |
| `transformer/diffusion_pytorch_model-00002-of-00002.safetensors` | 6263592868 | `7bed2b1d44fb6e69dd7db5b2f2ab1c54baf4ae6b0e3287b694c929f9f6192402` |
| `vae/diffusion_pytorch_model.safetensors` | 336213556 | `d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5` |

需要修复/补下时，在服务器执行 `bash /root/fibo2/download-models.sh`。它使用 `models.aria2` 断点续传，再逐项计算 SHA256，全部通过后才写完成标记；**不要手工创建完成标记**。清单、下载脚本和 aria2 输入已复制到本机审计目录，但模型本身没有经本机中转，也未下载到本机。

许可：模型卡声明非商业限制，HF 元数据为 **bria-fibo** 且仓库 gated；不要把镜像公开可下载理解为无条件商用授权。正文及元数据的许可链接存在差异，应以 BRIA 当前条款为准；正式使用前确认许可，**商用需 BRIA 单独协议**。本次没有替用户接受 HF 门禁条款、提交个人信息，也没有独立比对受限 HF 原文件的 SHA。

- 官方模型卡：<https://huggingface.co/briaai/Fibo-2.0-edit-turbo>
- 许可：<https://platform.bria.ai/legal/bria-fibo-license>
- 官方 pipeline 文档：<https://huggingface.co/docs/diffusers/main/api/pipelines/bria_fibo2>
- 已核对版本源码：<https://github.com/huggingface/diffusers/blob/v0.42.0/src/diffusers/pipelines/bria_fibo2/pipeline_bria_fibo2.py>

## 6. 启动、健康与停止

目前服务已运行，**不要重复启动**。查看状态：

```bash
curl -fsS http://127.0.0.1:8190/health
ps -p "$(cat /root/fibo2/service.pid)" -o pid,args
tail -n 30 /root/fibo2/log/service.log
```

重启服务器后需手动启动（没有注册开机自启）。仅在确认没有旧服务进程、8190 未占用时：

```bash
cd /root/fibo2
setsid nohup bash start-fibo2.sh >"log/service-$(date +%Y%m%d-%H%M%S).log" 2>&1 </dev/null &
echo $! >service.pid
# 等待模型加载，再检查 /health；READY 前不能推理。
```

停止前先确认 `/health` 的 `busy=false`、核对 `service.pid` 对应命令确为 `run-fibo2.py --serve --port 8190`，再对该 PID 执行 `kill`；不要 `pkill python`，也不要把文档里的旧 PID 当当前 PID。

本地/服务器脚本自检分别为：

```bash
uv run --no-project python run-fibo2.py --self-test
/root/fibo2-venv/bin/python /root/fibo2/run-fibo2.py --self-test
/root/fibo2-venv/bin/python -m pip check
```

## 7. 真实验收与本机审计

| 输入与任务 | 输出 | 服务端耗时 | NVML 采样峰值 | Torch allocated / reserved 峰值 |
|---|---|---:|---:|---:|
| 四图＋英文四格分镜指令 | 1024×1024 PNG | 67.927 s | 22546 MiB（22.02 GiB） | 18.547 / 21.695 GiB |
| 单图＋英文水彩指令，本机经 SSH 上传返回 | 768×768 PNG | 32.511 s | 17910 MiB | 16.142 / 17.168 GiB |
| 单图＋中文黑白素描指令，本机经 SSH 上传返回 | 768×768 PNG | 29.233 s | 17912 MiB | 16.140 / 17.170 GiB |

三次均恰好 4 步、成功输出、PNG 尺寸及 SHA256 一致，无 GPU OOM，容器 `oom/oom_kill=0`；检查图片有有效内容并符合分镜/水彩/素描指令。五种非法输入返回 400，忙碌时额外请求返回 503；客户端和服务器自检通过。两图/三图没有另做真实出图验收。

耗时从准备好的请求/输入落盘后开始，包含模型调用与输出编码，不含首次载入模型、网络上传/下载。NVML 每 0.5 秒采样，不是瞬时硬峰；三个测试各自的采样窗口依任务时间截取。GPU 利用率采样最大 100%，不代表全程满载；这些是单次验收数据，不是并发吞吐承诺。

公开输入复用服务器先前直接下载的 Blender Foundation/Sintel 预告片，来源 <https://media.w3.org/2010/05/sintel/trailer.mp4>，取片中约 31、34、38、36 秒，四张均 864×480、SHA 不同；第四张在服务器提取，没有图片发送到外部处理服务。

本机审计位置：

```text
/Users/yingtiankai/Desktop/Runbooks/video-generation/fibo2/
  tmp-fibo2-acceptance-20261031.bCLVVp/
    inputs/          # 原始 4 图、提示词、来源
    outputs/         # 四图结果、英文单图、中文单图 PNG 及客户端元数据
    evidence/        # 三个完整 job、验证报告、下载清单、脚本、日志和 GPU CSV
    SHA256SUMS       # 本机归档文件校验清单
```

该临时目录以 `.gitignore` 排除大文件，但保留在磁盘供审计。`evidence/deployment-verification.json` 汇总三次任务、版本、离线变量、回环监听、资源峰值和自检；`evidence/h3-model-cleanup.json` 精确记录删除的两项 H3 权重和恢复 URL。恢复 H3 需额外腾出约 34.14 GiB 再重新下载两项，当前余盘不足以直接让两套完整权重并存。

### `run.py` SSH 包装脚本验收

另已从本机实际执行一次完整 **SCP 上传四图 → SSH 生成 → SCP 下载**，输出 1024×1024、4 步、seed 43，服务端耗时 **64.859 秒**（不含 SSH 传输），job 为 `fibo2_20261031_143849_ee7c3207`。中文/空格/单引号文件名、带单引号的提示词均通过，下载 SHA256 与服务器原件一致，成功后仅本次中转目录被清理，原始 job 保留；没有更改或重启推理服务。

该次输入、输出、完整 job、测试日志和校验报告在本机 `tmp-fibo2-ssh-20261031.jX2Y81/`。四项单元测试覆盖上传/下载和 shell 引号、哈希损坏、推理失败不重试/不清理、已有文件与非法引用不连接服务器。

## 8. 多参考图角色替换配方（2026-10-31 新机实测）

> 提示词写法的通用规则（VGL JSON 骨架、只改该改的部分、动作与方位、配饰翻倍等）已抽到 [`skill.md`](skill.md)，本节保留本部署的实测数据与工作记录。

**先说失败模式。** 只有自然语言、两张图直接传时（job `fibo2_20261031_212013_a0209d78`），Fibo2 会把任务当成 multi-reference 重新构图：原图机位和姿态全丢，还把设定稿里的不同视角拼成两个人。必须用结构化提示把它拉回 in-place edit。

**配方**（三张成片都用它）：

1. 角色图裁成**单视角**（四视图设定稿裁出正面单人，320×820 即可）；整张多视图更容易串视角。
2. 画布放 `<image_1>`、角色放 `<image_2>`；宽高按画布比例取 16 的倍数（本次 1024×688 / 1024×640 / 720×1024）。
3. 提示词写成 **VGL JSON**：`structured_caption` 描述**结果图**，`edit_instruction` 描述**改什么**，英文。`run-fibo2.py` 原样透传，`BriaFibo2Pipeline._parse_prompt` 拆成 caption + instruction；服务器 `jobs/<job_id>/request.json` 的 `encoded_edit_prompt` 可核对解析结果（caption 键放顶层或 `structured_caption` 下都行）。
4. caption 写死机位、构图、姿态、在场人物与光源方向 —— 这是把模型按住原构图的关键；instruction 用「Change only … keep … exactly as in `<image_1>`」。
5. 模型卡声明该 checkpoint 只用英文训练；中文自然语言多图指令更容易被当成重新构图。

**关键一条：只改该改的部分。** 第一版让模型把角色整套服装穿到人物身上，成片很干净，但除了姿势之外几乎全部重画：原图的身体状态、与其他对象的接触关系、其他对象的细节都被覆盖掉了。用作视频首帧替换时这等于换了一场戏。第二版改成「只换头部（脸、发色、发型、发饰、瞳色），身体、皮肤、着装状态、姿势以及与画面中其他对象的关系全部按 `<image_1>` 保留，不覆盖、不移动、不添加」之后，三张一次过，耗时不变（约 30 s）。

骨架：

```json
{"short_description":"结果的画面：谁、什么姿态、在什么环境",
 "objects":[{"description":"<image_2> 角色的头（发型、发色、发饰、瞳色）+ 与 <image_1> 相同的身体和着装状态","location":"画面中部，具体姿态"},
            {"description":"次要人物/道具，写清画面里出现的是哪个部分","location":"左侧"}],
 "background_setting":"房间、床、窗、光源位置",
 "lighting":{"conditions":"柔和日间光","direction":"来自左侧"},
 "style_medium":"anime illustration, clean cel shading",
 "composition":{"camera_angle":"eye level","framing":"tight horizontal close-up"},
 "edit_instruction":"Change only the woman's head. Give her the face, hair color, hairstyle and hair accessories of the character in <image_2>. Keep her body, skin, clothing state, pose and the overlap with the other objects exactly as in <image_1>. Do not cover, move, add or remove anything. <image_2> is a head and hair reference only: do not copy its outfit, white background, standing pose or other views."}
```

| 版本 | 尺寸 | 画面耗时 | job |
|---|---|---:|---|
| task1 只换头（采用） | 1024×688 | 32.9 s | `fibo2_20261031_224742_6b1567da` |
| task2 只换头（采用） | 1024×640 | 29.9 s | `fibo2_20261031_215142_35594fa7` |
| task3 只换头（采用） | 720×1024 | 31.2 s | `fibo2_20261031_214255_67dd73af` |

`result/` 下 `result-taskN.png` 是采用的版本；task1 另留了三个对照：`result-task1-v1.png`（衣物把局部细节盖住了）、`result-task1-v2.png`（同 prompt seed 42，结构仍偏平）、`result-task1-v3.png`（seed 7 旧 prompt）。`result/.gitignore` 为 `*`，不进仓库。原始 job 与完整 prompt 在服务器 `/root/fibo2/jobs/<job_id>/`。

**关键结论：局部互动关系必须写实。** 只写"保持两者的关系不变"，Fibo2 会把互相接触的部分画成两块互不相干的形状，位置关系似是而非；把动作与方位写明（两者各自在做什么、谁位于画面哪一侧、朝向是什么、接触点大致落在画面什么位置），结构立刻变合理。同时保留"形状与位置以 `<image_1>` 为准"，避免它自由发挥。这是本次多次返工后唯一稳定的写法。

**只换头也不等于会自动保住每一个细节。** task1 第一版（`fibo2_20261031_214741_f1bbd5ba`，即 `result-task1-v1`）虽然只换头，模型还是把衣物重画到两者之间，把原本可见的部分盖住了。修法是在 caption 里写清这个位置、遮挡关系与可见范围，并在指令里明确禁止遮挡（不要遮挡、不要补画、不要移动衣物），构图和身份都不会因此回退。

第二个坑：**过度描述局部反而更糟**。把局部几何写得很细（逐项要求重叠方式、边缘形状）时，模型会把它画成一块和身体脱节的形状（job `fibo2_20261031_222037_9b220f28`），构图直接错。正确的粒度是：动作与方位写死，形状和面积交给 `<image_1>`。

第三个坑：**头发配饰会翻倍**。同一条指令跑出的两版里，一版在头两侧各画了一个铃铛蝴蝶结（`..._223439_2d156625`）。在 instruction 里补一句「one black bow with a gold bell on the left side; no second hair clip」后消失，且未影响其他部分。

最后确认 seed 只影响细节：同一版提示词在 seed 42 / 7 / 1234 下都能出合理结果，差别在局部细节的形态与配饰数量。采用 seed 42，与 task2/task3 保持一致；调 seed 是细节微调手段，不能替代把动作写清楚。

**仍有的瑕疵**：task2 把参考图领口的细节带成胸前的金色吊坠；task1 保留原姿势后右手手指略粗；人物肤色整体偏白（参考角色本身偏白）；Fibo2 会轻微压平原图柔和阴影。

**与 Qwen 部署的取舍**（不同机器、不同 GPU，只比调试手感和单张效果，不比较吞吐）：

- 耗时：Fibo2 4 步约 29–35 s；Qwen Turbo 8 步在旧机上约 110–160 s。
- 提示词形态：Fibo2 需要英文结构化 caption（JSON），Qwen 只需 `<image1>` 标签 + 职责句，中文直接可用。
- 失败表现：Fibo2 失败是「重新构图 / 串视角」，一眼能看出来；Qwen 失败是「几乎没改」，容易被当成成功。
- 局部保持：只换头这种任务两者都能做；Fibo2 少一层「模型自己发挥」的涂抹，线条更干净；Qwen 更贴原图的柔光与笔触。
- 参考图：Qwen 可整张吃四视图设定稿；Fibo2 建议先用单视角裁片。
- 两者都必须固定 seed 做单变量对照，才能判断哪句提示词起作用。

## 边界

这是 SSH 私用、单 GPU 串行服务，只监听回环地址，**没有公网鉴权/多租户隔离，不要直接暴露公网**。未做并发压力、长期稳定性、mask、训练、无图生成或 >1024 分辨率测试；官方自动 VGL 扩写未启用。商用许可需另行确认。
