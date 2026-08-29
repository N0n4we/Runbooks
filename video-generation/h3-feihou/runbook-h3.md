# MiniMax H3 FeiHou Remix 部署 Runbook

目标机：`root@connect.westb.seetacloud.com:27454`，NVIDIA GeForce RTX 4090
（该实例当前报告 48 GiB VRAM）。使用同一个同时支持
**FL2VA/REF2VA** 的 INT8 ConvRot Remix DiT，不下载独立的 REF2VA 权重；精度为
**INT8 ConvRot Remix DiT + INT8 uncensored Qwen3VL encoder**，并加载
**8-step Turbo LoRA**。

## 文件布局

本目录包含：

- `MiniMaxH3-FeiHou-Easy-H3.json`：页面指定的 FeiHou Easy H3 v2.0 工作流
  （可自动选择 FL2VA/REF2VA）；
- `run-h3.py`：上传首/末帧、转换工作流、远程提交和拉回 MP4；
- 本文件。

本次 AutoDL 适配中，实际 ComfyUI 路径为 `/root/ComfyUI`，并通过
`~/comfy/ComfyUI` 兼容本文件和脚本的固定路径；模型、输入、输出和启用的节点库
位于快速数据盘 `/root/autodl-tmp/h3-comfy/`，系统盘不存放大模型。镜像中原有的
非 H3 节点保留在 `custom_nodes-disabled/`，当前仅加载本工作流需要的四个节点库。

原方案已移入本地 `video-generation/h3-origin/`；目标机旧的 H3 Turbo/QwenVL
节点已移入 `~/comfy/h3-origin/custom_nodes/`，不再被 ComfyUI 加载。

当前 FeiHou 主节点有两个互斥的输入路径：`image` 路径是 FL2VA 首帧/首尾帧，
`reference` 路径是 REF2VA。REF2VA 路径本身支持混合图片、视频和独立音频；但
一次采样不能把 FL2VA 首/末帧和 REF2VA 参考媒体同时接入，因为主节点的
`mode` 是单选项，不是两个模型的并行混合图。`run-h3.py` 不暴露 `--mode`：
传 `--image/--last-image` 自动选 FL2VA，传 `--ref-image/--ref-video/--ref-audio`
自动选 REF2VA，同时传两类参数会明确报错。

## 必需模型

将文件放到目标机下列位置（文件名必须一致）：

| 路径 | 作用 |
|---|---|
| `models/diffusion_models/FeiHou_MiniMax-H3_Remix_v0.6_int8_convrot_v2.safetensors` | INT8 ConvRot Remix，共享 FL2VA/REF2VA |
| `models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot_uncensored.safetensors` | INT8 uncensored Qwen3VL |
| `models/vae/minimax_h3_video_vae_fp16.safetensors` | 视频 VAE |
| `models/vae/minimax_h3_audio_vae_fp32.safetensors` | 音频 VAE |
| `models/loras/H3/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | 8-step 推理加速 LoRA |

其中模型文件和 LoRA 的 Civitai 下载需要登录授权。目标机无法直连
`civitai.com`，但可以经 `civitai.red` 带授权下载代理直接下载。

页面指定的完整工作流是 Civitai workflow model version `3257904`
（file `3141087`，SHA256
`68CB76BA690C2B5D45144E20E17BE74C6ACF85D31B5ED65F440F98ED82494FCD`）。
当前目录中的 JSON 已按上述 SHA256 校验。`run-h3.py` 会按节点类型重新写入
模型、精度和 Turbo LoRA，不依赖固定节点编号；默认只运行第一采样以避免对
工作流中的额外二采模型产生隐式依赖，传 `--second-sampling` 时则把同一个
Remix 模型用于二采 FL2VA/REF2VA 路径。

## 镜像下载情况

目标机可以访问 ModelScope CDN；Turbo LoRA 已优先从官方镜像下载并校验
SHA256 `2339acdf19bfe123f46b971ea35d367a84adb85de43627e1eceafa5a5b2b111e`。
目标机无法稳定访问 Hugging Face mirror；已使用带授权的 `civitai.red` 代理在
服务器直下 Civitai 文件，不需要本地传输大模型。不要把普通 INT8 encoder 或
分开的 FL2VA/REF2VA 权重冒充 Remix 全套文件。

## 依赖节点库与启动

下面是**当前 JSON 工作流和 `run-h3.py` 实际需要的完整节点依赖**。不要只按
页面的 Required Nodes 安装：`VideoCombineV2` 还需要 VideoHelperSuite，二采样
还需要 KJNodes。

| 依赖 | 仓库 | 工作流中使用的节点/作用 | 必需性 |
|---|---|---|---|
| ComfyUI core | [Comfy-Org/ComfyUI](https://github.com/comfyanonymous/ComfyUI) | `BasicGuider`、`BasicScheduler`、`KSamplerSelect`、`SamplerCustomAdvanced`、`ModelAttentionBackend`、`VAEDecode/VAEEncode`、`VAEDecodeAudio/VAEEncodeAudio`、`ResolutionSelector`、`LTXVConcatAVLatent` | 必需 |
| ComfyUI-FeiHou-Easy-H3 | [FX-FeiHou/ComfyUI-FeiHou-Easy-H3](https://github.com/FX-FeiHou/ComfyUI-FeiHou-Easy-H3) | `FeiHouEasyH3Loader`、`FeiHouEasyH3`、`FeiHouEasyH3Output`、`FeiHouEasyH3LoraStack`、`FeiHouEasyH3PromptPreview` | 必需 |
| ComfyUI-FeiHou-Toolbox | [FX-FeiHou/ComfyUI-FeiHou-Toolbox](https://github.com/FX-FeiHou/ComfyUI-FeiHou-Toolbox) | `VideoCombineV2`、`RandomSeedNoise` | 必需 |
| ComfyUI-VideoHelperSuite | [Kosinkadink/ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) | `VideoCombineV2` 内部调用原生 `VHS_VideoCombine`，负责视频/音频封装 | 必需 |
| ComfyUI-KJNodes | [kijai/ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) | `ImageResizeKJv2`（二采样）；`SetNode/GetNode` 是该库提供的前端虚拟连线节点，转换 API workflow 时会变成实际连线 | 必需 |

说明：`ResolutionSelector`、`LTXVConcatAVLatent`、`MarkdownNote` 属于 ComfyUI
本体（后两者分别是 core extra/UI 节点），不是额外节点库。JSON 中
`ImageResizeKJv2` 的旧 workflow 元数据可能显示 `aining2022/ComfyUI_Swwan`，
但目标机实际注册和运行它的是 **ComfyUI-KJNodes**，不需要另装 Swwan。

以下是已安装但**不是当前工作流推理依赖**的包：

- `ComfyUI-Easy-Use`：当前 JSON 没有 Easy-Use 节点，可保留但不应列为必装依赖。
- `rgthree-comfy`：原始 JSON 的 `Label (rgthree)` 和 `Fast Groups Bypasser (rgthree)`
  仅用于画布标注/旁路 UI。`run-h3.py` 转 API prompt 时会删除它们，因此 CLI 推理
  不需要；若要在 ComfyUI GUI 中无缺节点地查看原始 JSON，可额外安装
  [rgthree-comfy](https://github.com/rgthree/rgthree-comfy)。

除节点库外，VideoHelperSuite/Toolbox 运行需要服务器可执行的 `ffmpeg`；本地
`run-h3.py` 的输出门禁还会使用本机 `ffprobe`（没有时只跳过本地门禁，不影响远端
推理）。

当前目标机基础环境：

- ComfyUI `0.34.0`；
- Python 环境 `~/h3-venv`（uv 创建，实际位于数据盘）；
- PyTorch `2.12.1+cu130`，CUDA 13.0，NVIDIA GeForce RTX 4090；
- Triton `3.7.1`；
- `comfy-cli` `1.20.0`；`comfy-kitchen` `0.2.32`（0.34.0 的 INT8 attention API 所需）；
- `ComfyUI-FeiHou-Easy-H3`、`ComfyUI-FeiHou-Toolbox`、`ComfyUI-VideoHelperSuite`、
  `ComfyUI-KJNodes`；镜像中其他节点库保留在 `custom_nodes-disabled/`，默认不加载。

目标机的 FeiHou 节点保留完整 H3 block streaming；由于当前 ComfyUI 0.34.0
的 `MiniMaxH3Model.final_layer` 签名已变化，节点中的旧版 output-head streaming
被关闭（`final_layer_chunk=0`），改用当前后端原生 output head，避免二采样时的
兼容性错误。

启动（脚本会自动启动，也可手动执行）：

```bash
ssh -p 31960 CUYvwa@wp08.unicorn.org.cn
cd ~/comfy/ComfyUI
export PATH=$HOME/h3-venv/bin:$HOME/.local/bin:$PATH
export XDG_CACHE_HOME=$HOME/autodl-tmp/h3-comfy/cache
setsid nohup python main.py --listen 127.0.0.1 --port 8188 \
  --disable-auto-launch --reserve-vram 4 \
  >/tmp/comfy_boot.log 2>&1 </dev/null & disown
curl -fsS http://127.0.0.1:8188/object_info >/tmp/object_info.json
```

确认 `/object_info` 中存在 `FeiHouEasyH3Loader`、`FeiHouEasyH3`、
`FeiHouEasyH3LoraStack`、`VAEDecodeAudio`、`ImageResizeKJv2` 和
`VideoCombineV2`。`GetNode`/`SetNode` 以及 `MarkdownNote` 不出现在
`/object_info` 是正常的：前两者是 KJNodes 前端节点，后者是画布 UI 节点。

## 执行与验证

在本机 Runbooks 根目录执行：

```bash
python3 video-generation/h3-feihou/run-h3.py \
  --prompt 'A person walks slowly through a sunlit garden, cinematic camera movement.'

# 首尾帧 FL2VA（自动选择 image 路径）
python3 video-generation/h3-feihou/run-h3.py \
  --prompt 'The subject moves naturally while the camera slowly pushes in.' \
  --image first.png --last-image last.png --seconds 5 --seed 42

# 可选：在同一个 Remix 上启用页面 v2.0 的二次采样（12 步首采样 + 4 步细化）
python3 video-generation/h3-feihou/run-h3.py \
  --prompt 'A short cinematic motion test.' --image first.png \
  --second-sampling --seconds 5 --seed 42

# REF2VA：可按需混合图片、视频和独立音频（图片最多9、视频最多3、音频最多3）
python3 video-generation/h3-feihou/run-h3.py \
  --prompt 'The person in <Picture 1> enters the scene in <Picture 2>.' \
  --ref-image person.png --ref-image scene.png --ref-video motion.mp4 \
  --ref-audio ambience.wav --seconds 5 --seed 42

# 仅做本地结构检查，不连接目标机、不需要模型
python3 video-generation/h3-feihou/run-h3.py \
  --dry-run --prompt 'test' --seconds 5 --seed 42
```

默认使用 480P、16:9、24 fps、5 秒（自动吸附到 124 帧），使用页面工作流的
12 步首采样；`--second-sampling` 额外使用工作流的 4 步、0.2 denoise 二采，
并强制开启 FeiHou 的 force-offload 和低显存分块以给 INT8 DiT 留出显存。
ComfyUI 进程额外保留 4 GiB VRAM；VAE 常驻 GPU（2026-08-29 起不再使用
`--cpu-vae`）。FeiHou 节点在首次采样前会自动把 CLIP/VAE 逐出显存，
force-offload 在采样后卸载 DiT，因此 GPU VAE 解码在 32 GiB 卡上安全，
且对长序列是决定性优化：实测 10 s/243 帧 @480x704 单段，`--cpu-vae` 时
CPU 解码占 ~45 min（采样仅 ~2.5 min，单段合计 ~53 min），改 GPU VAE 后
单段约 4 min。冷启动后的第一个任务可能在采样器 OOM 一次：ComfyUI 显存
管理器会自适应加大 DiT 卸载，同一任务直接重跑即可成功。若 REF2VA 混合
媒体在 VAE 编码阶段 OOM，可临时退回 `--cpu-vae`（慢但稳）。Turbo LoRA
默认 strength 为 0.75。二采样分支原工作流的 NVIDIA VSR 在目标机自动降级为
Lanczos（目标机未安装 nvvfx），不影响工作流执行。`--auto-prompt` 只有在
ComfyUI FeiHou 设置中已经配置提示词 API 后才使用；默认不调用外部 API。

脚本会检查五个模型、隔离 run ID、以 `H3_<run_id>` 命名输出，并用 `ffprobe`
检查 H.264 视频和原生音轨。`--no-download` 可只在服务器生成而不拉回文件。

## 下载授权后的验收顺序

```bash
find ~/comfy/ComfyUI/models/{diffusion_models,text_encoders,vae,loras} \
  -type f -iname '*h3*' -o -iname '*remix*'
python3 video-generation/h3-feihou/run-h3.py --dry-run --prompt test
python3 video-generation/h3-feihou/run-h3.py \
  --prompt 'A short cinematic motion test.' --seconds 5 --seed 42
```

验收日志应显示四个模型文件（一个 Remix 同时供 FL2VA/REF2VA）和 Turbo LoRA
均成功加载；单采样使用页面的 12 steps，启用二采时另有 4 steps，并且最终
MP4 同时包含 video 和 audio stream。

## 提示词变量组合必须先 dry-run 实测

链式/批量脚本普遍用变量组合提示词（角色锚点、地点、镜头档、尾帧句式等）。
**规则：靠人工通读全文来判断提示词有没有问题。**
任何一次实际提交推理之前，必须先查看真正传入模型的最终拼接文本，
逐段逐句读完整拼接后的全文再发射；断言/自动检查只能作为辅助，
不能代替人工通读。禁止直接信任模板代码。

正确姿势（`workspace/chain-cctv.py` 已内置此模式）：

1. **判断提示词好坏的唯一可靠方法是人工通读完整拼接文本**。
   断言/自动检查只能辅助——断言本身会写错、会漏掉语法断裂和语义矛盾，
   多次实测中真正抓到问题的是通读，不是断言。
2. **脚本在提交前打印完整 PROMPT**（`print("PROMPT:", prompt, flush=True)`），
   日志里永远可回溯每一段实际用的提示词。
3. **发射前 dry-run 导出全部段落**：用 `importlib` 加载脚本、遍历 segments
   打印最终文本，逐段逐句读（54 段级别的批量也要全读），重点读变量嵌入处
   前后各一句话。
4. **断言只作为回归辅助**：把本次通读发现的问题固化成断言，防止下次回归；
   但断言通过 ≠ 可以发射，必须先完成人工通读。
5. **未完成通读就发射属于流程违规**（实测发生过：先发射后审查，
   被迫中断重跑）。

逐段检查清单（历史上每个条目都真实踩过坑）：

- [ ] **变量嵌入处语法完整**：锚点含逗号/定语时嵌入模板会断裂
- [ ] **人物在场状态与镜头描述一致**
- [ ] **角色/场景锚点逐段重复**且用词一致（模型无跨段记忆）
- [ ] **视角锚点每段开头重复**（监控固定机位/POV"主角脸全程不可见"等）
- [ ] **每段结尾的尾帧描述与下一段首帧衔接**，且无双重/矛盾指令
- [ ] **指代词无歧义**（"她/他"多个候选时改用完整称谓）
- [ ] **尺寸/负载**：画布对齐首帧长宽比，负载 ≤ ~83M px·frames（48GB 机器
      实测 124M 可用），尾帧提取后按 run 尺寸校验
