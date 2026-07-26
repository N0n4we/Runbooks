# Wan2.2 官方 I2V 图生视频 — 部署与执行 Runbook

> 目标机：wp08.unicorn.org.cn:13988（Lt2s9y）/ RTX 4090 24G / ComfyUI `~/comfy/ComfyUI`（8188，常驻 `--lowvram`）
> 底模：官方 Wan2.2-I2V-A14B（fp8_scaled / GGUF Q8_0 双备）；底图：Comfy-Org 官方极简模板
> 一键脚本：`run_wan.py`；本 runbook 只记方法与经验，不记录任何私人 LoRA 文件名。

## 0. 这是什么 / 底图沿革（重要教训）

- 管线 = 官方 Wan2.2 I2V 双专家（高/低噪两个 14B）+ 核心 WanImageToVideo（36 通道 i2v 条件）+ `run_wan.py`。
- **底图现用 Comfy-Org 官方极简模板**（`Wan图生视频工作流.json`，19 节点单子图）。
- **Kenpechi SVI v3.5 已弃用**：其 WanAdvancedI2V（FMLF 包）被 Kenpechi 配为 `long_video_mode='SVI'`，该模式构建**小数掩码的 SVI 风格条件**（源码 wan_advanced_i2v.py:254-267），必须配 SVI PRO LoRA 才能被模型读懂；适配时 SVI PRO 被旁路（文件不在服务器）→ 底模裸接分布外条件 → 「首帧正常→瞬间混沌」（T-A~T-E 对照定位）。当时的另一条活路是把 mode 改 DISABLED（有纯 i2v 路径）；最终选择官方模板重建，更干净。教训：拆“配套 LoRA 非必需”的判断要看条件构造方式，不能只看段数。

## 1. 环境前提

- 服务器已按 runbook-bernini.md 部署（ComfyUI 当前版、umt5、wan_2.1 VAE 等在位）。
- 节点包：仅 `city96/ComfyUI-GGUF`（`pip install gguf`，装完重启 ComfyUI）。FMLF 包已装但**不再使用**（留待上游修复）。
- github 直连不稳时的 zip 搬运法、节点注册验证（**必须解析 `/object_info/<name>` 的 body**，urlopen 不报错 ≠ 节点存在）：

```bash
curl -L -x http://127.0.0.1:7890 -o /tmp/pack.zip https://codeload.github.com/<owner>/<repo>/zip/refs/heads/main
scp -P 13988 /tmp/pack.zip Lt2s9y@wp08.unicorn.org.cn:/tmp/
ssh -p 13988 Lt2s9y@wp08.unicorn.org.cn 'cd ~/comfy/ComfyUI/custom_nodes && \
  python3 -c "import zipfile; zipfile.ZipFile(\"/tmp/pack.zip\").extractall(\"/tmp/\")" && \
  mv /tmp/<repo>-main <repo-name>'
```

## 2. 模型文件（`models/diffusion_models/`）

| 文件 | 大小 | 用途 | 来源（实测） |
|---|---|---|---|
| `wan2.2_i2v_{high,low}_noise_14B_fp8_scaled.safetensors` | 各 13.3G | 默认底模（不挂 LoRA） | ModelScope `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` |
| `wan2.2_i2v_{high,low}_noise_14B_Q8_0.gguf` | 各 14.35G | 挂 LoRA 必选 | ModelScope `bullerwins/Wan2.2-I2V-A14B-GGUF` |

- **fp8_scaled + LoRA = 死局**：lowvram 补丁合并后每层随机舍入重量化回 fp8（`stochastic_rounding_fp8→calc_mantissa`），临时变量成堆，有无 `--lowvram` 都 OOM。脚本检测到挂 LoRA 会自动切 gguf。
- T5 用服务器原有 umt5（fp16 内容），CLIPLoader 设备保持 `default`（GPU）。
- 加速对（`models/loras/`，2026-07-26 下载）：`wan2.2_i2v_lightx2v_4steps_lora_v1_{high,low}_noise.safetensors`（各 1.14G，ModelScope `Comfy-Org/Wan_2.2_ComfyUI_Repackaged`）。**用法：`--steps 4 --split 2 --cfg 1,1` + 高/低链各挂一只（蒸馏模型强制 cfg 1）**，脚本自动切 GGUF。已实测 4 步出片（T-acc）。

## 3. 工作流结构（`Wan图生视频工作流.json`）

```
LoadImage(97) ─┐
               ▼
┌─ 子图 d2ac71a3「Image to Video (Wan 2.2)」(实例 130) ─────────────┐
│ UNETLoader 122/123(fp8|GGUF) → LoraLoaderModelOnly 126/127(槽位)   │
│   → [串联插入的 LoraLoaderModelOnly…] → ModelSamplingSD3 109/124   │
│   (shift=5) → KSamplerAdvanced 110(高:0→split,加噪) /              │
│   111(低:split→end,接力)                                          │
│ CLIPLoader 105 → CLIPTextEncode 107(正)/125(负)                    │
│ VAELoader 106 → WanImageToVideo 128(36ch i2v 条件) → 采样器        │
│ VAEDecode 129 → CreateVideo 117(fps)                               │
└──────────────────────────────────────────────────────────────────┘
               ▼
        SaveVideo(108) → output/*.mp4
```

- 子图实例 130 用 proxyWidgets，值直接补丁在**内部节点 widgets** 上（130 自身 `widgets_values` 为空）。
- 采样器：euler + simple，高噪 add_noise=enable、低噪 disable 接力；cfg 默认 4,4。

## 4. 一键脚本 run_wan.py

```bash
python3 run_wan.py --image 图.png --prompt "……" --seconds 3
python3 run_wan.py --image 图.png \
    --lora-high A.safetensors:0.6 B.safetensors:0.5 \
    --lora-low  A_l.safetensors:0.6 B_l.safetensors:0.5   # 自动切 gguf
# 参数：--size 480x640（×16）/ --seconds 3 / --fps 16 / --steps 12 / --split 6
#       --cfg H,L（默认 4,4）/ --seed N（缺省随机）
#       --base fp8|gguf（默认 fp8；挂 LoRA 自动 gguf）/ --unet-dtype / --no-download
```

行为：上传输入图 → 内存补丁共享 json（不动本地文件；LoRA 第 2 只起自动串联插入节点）→ `comfy run` 提交 → run_id 隔离轮询 → 拉回 ffprobe 验证。输出 h264 mp4（QuickTime 可播）。

## 5. 24G 显存与画质经验（2026-07-25~26）

- 无加速：12 步切 6、cfg 4,4 出正常画面；更高质量 20~24 步。加速 LoRA（lightx2v 官方 4 步对）文件未传服务器，模板 126/127 槽位现作用户 LoRA 首槽。
- GGUF Q8_0 + 三对 LoRA（高/低链各 3 只）480×640×49f 一次通过；fp8+LoRA 见 §2 死局。
- 换 LoRA 配置连跑易撞残留 OOM，**换配置前重启 ComfyUI 最稳**；重启用两条 ssh（pkill 与启动写同一条会被 pkill -f 自匹配杀掉，exit 255）。
- 服务器 torch 2.7.0.dev（<2.8）：ComfyUI 回退 legacy ModelPatcher，官方核心节点全部正常，但第三方重型补丁节点（FMLF）会坏——装新节点包后先用极简图验证再上大图。

## 6. 验证记录（wp08 / 4090 24G / --lowvram，2026-07-26）

| 项 | 值 |
|---|---|
| T-E 基准 | 官方模板 + fp8 + 无 LoRA + 12/6 + cfg 4,4 + 49f@480×640 → ✅ 正常画面 |
| T-F | run_wan.py 默认（=T-E 配置）→ ✅ |
| T-G | run_wan.py + GGUF + 高/低链各 3 只 LoRA（0.5~0.6，1 内置槽 + 2 串联）→ ✅ 通过 |

## 7. 故障速查

| 症状 | 原因/处理 |
|---|---|
| 首帧正常→瞬间混沌 | 若底图含 FMLF WanAdvancedI2V → 弃用换官方模板（本服务器不兼容）；否则查 sigma/条件链 |
| OOM 栈落在 `stochastic_rounding_fp8/calc_mantissa` | fp8_scaled 挂了 LoRA → `--base gguf`（脚本已自动切） |
| 校验 `class_type 'X' not found` | 缺节点包（§1）；自查 object_info 要解析 body |
| 校验 `Required input is missing: model`（shift 节点） | 串联插入时改了 `sg["links"]` 的**副本**没改列表——`chain_lora` 已修复；同类手术都直接操作列表 |
| ssh 执行到 pkill 后断（exit 255） | kill 与启动须分两条 ssh（pkill -f 会匹配同串命令行里的 "main.py" 字样自杀） |
| 脚本轮询永不退出（任务其实早完成） | pgrep/pkill 的模式串会被**远端 bash -c 自身命令行**匹配：模式必须用方括号打断自匹配（`comfy r[u]n --workflow`、`[m]ain.py`） |
| 拉回的 mp4 打不开/缺 moov | 旧版按“文件出现”就 scp 的竞态（h264 moov 最后落盘）；现版轮询 comfy run 进程退出 + 本地 ffprobe 门禁。服务器 output/ 里的原件一直完整，坏的是本地截断副本，可按 run_id 重拉 |
| 视频打不开 | 输出已是 h264 mp4；若手动改过 SaveVideo 格式，QuickTime 只认 h264/h265 mp4 |
