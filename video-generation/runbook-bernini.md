# Bernini-R 图生视频工作流 — 部署与执行 Runbook

> 目标：在已有 SCAIL-2 部署的 GPU 服务器上，把 ByteDance **Bernini-R**（Wan2.2-T2V-A14B 架构，high/low 双模型两段采样）图生视频工作流部署跑通。
> 本 runbook 基于 **wp08:13988（RTX 4090 24G / 503G RAM / cgroup 128G）** 实测（2026-07-25），该机器已按 runbook-scail.md 完成 ComfyUI 基础环境（master + 8 节点包 + libgl/ffmpeg + sageattention）。
> **加速 LoRA 一对（high noise / low noise）由用户自备**（见第 6 节），缺失时 bypass 验证，管线其余部分已全量实测通过。本 runbook 只记录方法，不记录任何实际使用的 LoRA 文件名。

---

## 0. 工作流是什么

`Bernini图生视频工作流.json`（本地仓库）= Wan2.2 **Bernini-R** 图生视频：

- **核心节点 `BerniniStudio`**（id=27，来自 CCpt5/ComfyUI-BerniniStudio）：一个节点包办 T5 文本编码 + VAE 编码参考图/源视频 + 条件构建（替代 5+ 个节点的连线）。输入 clip/vae/image0~7/source_video，输出 positive/negative/latent。
- **双 UNET 两段采样**：`wan2.2_bernini_r_high_noise_int8_convrot`（前段 sigma）→ `wan2.2_bernini_r_low_noise_int8_convrot`（后段），stock `SamplerCustom`×2 + `SplitSigmas`，`ModelSamplingSD3`，euler。
- **4 步快采**：BasicScheduler `["simple", 4, 1]` + SplitSigmas `[2]`（高/低各 2 步）。该节奏配合蒸馏加速系 LoRA（用户自备）；不挂 LoRA 时 4 步画质粗，底模出片建议 24 步 + cfg 4,4。
- **画布/帧数控制链**：INTConstant 34(width) / 35(height) → SetNode → GetNode → 27；`Int` 31 → SimpleMath+ `a*16+1` → length。31 填 5 → length=81（Wan 4n+1 网格）。**改 31 的数即可调帧数**（n → 16n+1 帧）。
- 输出：VHS_VideoCombine，16fps，`output/%date:...%_Bernini_00001.mp4` + 首帧 png。

## 1. 环境前提

复用 SCAIL 部署（runbook-scail.md 第 1-3 节）：ComfyUI master、comfy-cli、ffmpeg/libgl、sageattention、8 个节点包（其中本工作流直接依赖：VideoHelperSuite、rgthree、essentials、Custom-Scripts、KJNodes 的 INTConstant/GetNode/SetNode）。

Bernini 追加 2 个节点包（ghfast tar 法，同 SCAIL 第 3 节）：

```bash
cd ~/comfy/ComfyUI/custom_nodes
G="https://ghfast.top/https://github.com"
for repo in "CCpt5/ComfyUI-BerniniStudio" "yawiii/ComfyUI-Prompt-Assistant"; do
  d=$(basename $repo)
  wget -q --tries=2 "$G/$repo/archive/refs/heads/main.tar.gz" -O "$d.tar.gz" && gzip -t "$d.tar.gz" && {
    tar xzf "$d.tar.gz" && mv "$d-main" "$d" && rm "$d.tar.gz" && echo "OK $d"
  } || echo "FAIL $d"
done
pip install -r ComfyUI-Prompt-Assistant/requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --retries 20 --timeout 60
```

- `BerniniStudio` 只依赖 torch+comfy（无第三方 pip）；其 Ollama 自动提示词增强是**可选**，工作流里把 `auto_enhance` 设 false 即无依赖（见第 3 节）。
- `ImageCaptionNode`（图像反推提示词，VLM/API 系）来自 Prompt-Assistant，注册名 `node_id="ImageCaptionNode"`。工作流里它 mode=4 且只接 ShowText 展示，**装包仅为注册类型，不需要配 API/模型**。
- 重启 ComfyUI 后 `/object_info` 应见 `BerniniStudio` / `ImageCaptionNode`（总节点 1411）。

## 2. 模型（全部实测来源；~29G 新增）

| 目标路径 | 来源 | 大小 | 备注 |
|---|---|---:|---|
| `diffusion_models/wan2.2_bernini_r_high_noise_int8_convrot.safetensors` | ModelScope `Comfy-Org/Bernini-R`（与 HF 逐字节一致） | 14,535,868,680 | **首选 ModelScope**（~13MB/s 支持 Range）；HF `Comfy-Org/Bernini-R` 同路径 |
| `diffusion_models/wan2.2_bernini_r_low_noise_int8_convrot.safetensors` | 同上 | 14,535,868,680 | |
| `text_encoders/umt5_xxl_fp16.safetensors` | **复用 SCAIL 的 fp16 T5**：`os.link("umt5-xxl-enc-fp8_e4m3fn.safetensors", "umt5_xxl_fp16.safetensors")`（该机 fp8 名文件内容即 Wan repack fp16，见 runbook-scail 第 14 节） | 11,366,399,385 | 硬链接零下载；否则 ModelScope `Comfy-Org/Wan_2.1_ComfyUI_repackaged` `split_files/text_encoders/umt5_xxl_fp16.safetensors` |
| `vae/wan_2.1_vae.safetensors` | **复用**：`os.link("vae/Wan2_1_VAE_bf16.safetensors", "vae/wan_2.1_vae.safetensors")` | 253,815,318 | 同一文件两个名字 |
| `loras/<高噪 LoRA>.safetensors` | **用户自备**（scp 上传 loras/ 根目录） | — | 节点 7 |
| `loras/<低噪 LoRA>.safetensors` | 同上 | — | 节点 10 |

> LoRA 一律**裸文件名放 loras/ 根目录**：作者工作流里写的是 Windows 反斜杠子路径，Linux 文件枚举匹配不上。具体用哪对文件不录入本文档。

下载用 Range 断点续传脚本（ModelScope URL 格式 `/api/v1/models/{repo}/repo?Revision=master&FilePath={path}`，支持 206）。下完 `safe_open` 逐个校验。

## 3. 工作流 json 服务端兼容补丁（必须改，原样直跑会失败）

在**服务器副本**（`~/bernini_workflow.json`）上改，本地共享 json 不动：

1. **虚拟节点映射**（同 SCAIL 第 4 节套路）：`Int`(31) → `PrimitiveInt` 且**值必须是整数 `[5]` 不是字符串 `["5"]`**（否则 `shape_mismatch: expected INT, got str`）；`Label (rgthree)`×4（65/66/67/69，无连线纯装饰）→ 直接删节点；`Fast Groups Bypasser (rgthree)`(82) → `FastGroupsBypassSwitch`（无连线，保留 mode）。
2. **LoRA bypass**（用户未提供前）：节点 7/10 `mode=4`。bypass 时 LoraLoaderModelOnly 直通 MODEL，链路不断。
3. **启用 image0**：节点 26 LoadImage `mode=0`，文件名改成 `input/` 里实际存在的图（本轮复用 SCAIL 的 `06_left.png`）。作者原写的 `vol-01-05.png`/`0000.png` 均私有文件，25/36/76 保持 mode=4；VHS_LoadVideo 21（蜘蛛侠.mp4）保持 mode=4。
4. **i2v 任务化**：27 的 widgets[4] `task_type` `"t2v"→"i2v"`；widgets[5] prompt 换成图生视频运动描述；**widgets[9] `auto_enhance` 必须 false**（true 会调 127.0.0.1:11434 的 Ollama，服务器没有 → 卡死/报错）。
5. **画布**：34/35 从 1280×720 改 **640×864**（竖版匹配测试图比例 896×1200≈0.747 vs 640/864≈0.741；像素量 60% 降 VRAM）。横版图就保持 1280×720。

> **画布策略（2026-07-25 终版）**：脚本不做任何自动适配——**画布默认 480×640**（`--size` 覆盖，须 16 倍数）；帧数 `--frames`（4k+1）或 `--seconds`（按 fps 换算吸附 4k+1）；输出帧率 `--fps` 默认 17。能跑多大凭经验：**24G 挂 LoRA 上限 ≈ 36M 像素帧，32G ≈ 75M**（5.5 节实测表就是经验库），脚本只打印当前配置的像素帧数供判断，超了也不拦；OOM 了就重启服务降配再跑。不改动共享 json 的 Get/Set 链（绕开 bypass 时 Get/Set 解析的脆弱性）。共享 json 的 34/35 保留作者默认 1280×720。
6. **服务端兼容自检**：无残留 `Int`/`Label`/`Fast Groups Bypasser`/`Image Blank` 类型。

## 4. 启动与执行

```bash
cd ~/comfy/ComfyUI
setsid nohup python3 main.py --listen 0.0.0.0 --port 8188 --disable-auto-launch > /tmp/comfy_boot.log 2>&1 </dev/null & disown
# 等 API 起（/object_info 200）
export PATH=$HOME/.local/bin:$PATH
setsid nohup comfy run --workflow ~/bernini_workflow.json --host 127.0.0.1 --port 8188 --wait --verbose --timeout 7200 > /tmp/bernini_run.log 2>&1 </dev/null & disown
```

产物：`~/comfy/ComfyUI/output/%date:yyyy-MM-dd%/%date:...%_Bernini_00001.mp4`（目录名是 literal 占位符，同 SCAIL）。

## 5. 本轮成功验证记录（wp08:13988 / 4090 24G，2026-07-25）

- 配置：i2v、640×864、81 帧、4 步（高 2 + 低 2）、LoRA bypass、image0=06_left.png、auto_enhance=false。
- **成功**：`Prompt executed in 280.25s`（4.7 分钟）。输出 640×864、81 帧、16fps、5.06s，帧均值 Y≈219-223 且帧间 checksum 相异（非黑帧/定格）。
- 显存峰值约 14.6G（int8 双模型轮换加载 + 640×864 激活）；cgroup 峰值 70.6G（T5 fp16 11.4G + 双 int8 14.5G×2 都会进 RAM 缓存）。
- 已拉回本地：`~/Downloads/bernini_i2v_validation_20260724.mp4`。

## 5.5 带 LoRA 验收记录（wp08:13988 / 4090 24G / --lowvram，2026-07-25）

**结论：int8_convrot + LoRA 能跑，但必须 `--lowvram` 启动，且 24G 有明确规格天花板。**

| 配置 | 结果 | 耗时 |
|---|---|---|
| 640×864 × 81f（n=5）+ LoRA | ❌ OOM（node 15 High Sampler，`int8_linear→torch.cat` 瞬时分配；普通模式和 --lowvram 都挂） | — |
| 720×960 / 816×1104 + LoRA | ❌ OOM（画布更大更没戏） | — |
| 640×864 × 65f（n=4）+ LoRA | ✅ 194s | 峰值未测，稳定 |
| 640×864 × 49f（n=3）+ LoRA | ✅ 173.9s | |
| 512×688 × 81f（n=5）+ LoRA | ✅ 157s | 长视频选这个 |

- 像素×帧数预算是硬约束：65f@640×864 ≈ 36M 像素帧 ≈ 上限；81f 想跑就降到 512×688（28.5M）。
- 无 LoRA 时 640×864×81f 能跑（14.6G 峰值）；挂 LoRA 后高噪模型 = int8 14.5G + bf16 LoRA 3.7G，瞬时缓冲一顶就 OOM。
- **服务器 ComfyUI 现以 `--lowvram` 常驻**（Bernini+LoRA 必需；SCAIL 实测兼容）。
- 验收产物（本地 ~/Downloads/）：`bernini_lora_65f.mp4`（640×864×65f×16fps×4.06s）、`bernini_lora_81f.mp4`（512×688×81f×16fps×5.06s）、`bernini_lora_49f.mp4`。

## 6. 添加 / 更换 LoRA 的方法

**原则：默认底模，LoRA 全部由 `run_bernini.py` 参数指定，不改共享 json、不改脚本。**

1. **上传**：`scp -P 13988 文件.safetensors Lt2s9y@wp08.unicorn.org.cn:~/comfy/ComfyUI/models/loras/`（裸文件名，勿带子目录）。
2. **成对使用**（Bernini/Wan 是 high/low 双模型，LoRA 通常也成对）：
   ```bash
   python3 run_bernini.py --image 图.png --lora-high <高噪文件> --lora-low <低噪文件>
   # 强度：--lora-strength 0.8（默认 1.0，两个一起调）
   ```
   脚本自动：节点 7/10 解除 bypass + 写入文件名 + 校验服务器上文件存在。不挂时保持 mode=4（底模直跑）。
3. **追加更多 LoRA**（角色/风格类叠加在既有链上）：`--extra-lora 文件:0.6` 双链都挂；`--extra-lora 文件:0.6:high` / `:low` 只挂对应链（成对 LoRA 必须分链各挂一条）；可重复。脚本在 7→46、10→48 之间动态插 LoraLoaderModelOnly 节点。

**24G 堆叠实测终论**（2026-07-25，多轮含干净进程对照）：
- 小型 LoRA 多叠（双链各 3 只、总权重 ~1.8G，不挂加速对）✅ 通过；此时建议 `--steps 24 --cfg 4,4` 补质量；
- **大体量加速对 + 任何第三方 LoRA 同图 ❌**：死点固定在高噪跑完、低噪链加载时，与画布、--reserve-vram、进程新旧均无关——两链的补丁克隆在整张图执行期同时被引用，显存挤不出去。**加速对压成 fp8（体积减半）也解不开**：实测 fp8加速对 + 一只小型同链照死低噪链；而 3 只小型/链（无加速对）却能过——该组合本身在 int8 量化路径上触发整层物化，与 LoRA 总重无关；
- **fp8 LoRA 画质警示**：同种子 A/B（bf16 vs fp8 加速对）Y-SSIM 仅 0.61——4 步快采把微小权重噪声放大成明显不同的成片。fp8 省显存可用，但出片与 bf16 不同（好坏需肉眼判）；要稳定复现就别混精度；
- 同配置连跑没问题；换 LoRA 配置连跑易撞残留 → **换配置前重启 ComfyUI 最稳**；
- 想「加速对 + 风格 LoRA」兼得：上 32G，或把风格 LoRA 离线合并进底模再说。
4. **兼容性**：Wan2.1/2.2 14B 系 LoRA 基本可加载（缺键仅警告）；加速系 LoRA 配低步数低 cfg（1.5/1），非加速场景抬 `--steps 24 --cfg 4,4`。
5. 手动改法（不推荐）：服务器副本 json 节点 7/10 `mode=0` + widgets[0] 改裸文件名。

## 6.5 一键调度脚本 run_bernini.py

与 `run_scail.py`（原 scail_run.py，已一并改名）同风格：

```bash
python3 run_bernini.py --image ~/Documents/Games/input_maruko_1.png --out ~/Downloads/out.mp4
# 常用参数：--prompt / --task i2v / --frames 81 或 --seconds 5 / --size 640x864
#          --fps 16 / --steps 4 --split 2 --seed N --cfg H,L
#          --lora-high A.safetensors B.safetensors:0.6   （高噪链，可多个，:强度 缺省 1.0）
#          --lora-low  A.safetensors B.safetensors:0.6   （低噪链；第 2 个起自动串联插节点）
```

行为：上传输入图 → 内存中给共享 json 打补丁（虚拟节点映射/LoRA/画布/帧数/auto_enhance=false）→ 服务器提交 → 轮询拉回 → ffprobe 验证。默认配置 480×640×81帧@17fps（4.76s）。**脚本默认服务器 Lt2s9y@13988**。

**多任务并发安全**（2026-07-25 修复并实测）：每 run 生成唯一 `run_id`，输入文件名、输出 filename_prefix（`..._Bernini_<run_id>_00001.mp4`）、轮询 glob、/tmp 中转名全部带它——多开实例各取各的产物，不会串号；服务器串行排队执行。跑完自动清远端输入图。修前缺陷：固定输入名会被后任务覆盖 + 轮询「任何最新 Bernini 文件」会双抓同一份。

## 6.7 身份一致性调参（i2v）

排查过的非因素：cfg 默认 1.5/1 本就是蒸馏加速 LoRA 配套值；seed 接线正常（29→两个采样器）；`ref_max_size`（默认 848）只影响 r2v 参考流，**对 i2v 基本无关**（参考图本来就被缩进画布，画布长边 ≤848 时它什么都不做）。

真正有效的杠杆（按效果排序）：
1. **步数**：4 步快采结构锚定仓促 → `--steps 8 --split 4`（2倍时间）；上限方案 `--no-lora --steps 24 --cfg 4,4`（~6倍时间，无 LoRA 时要抬 cfg，默认 1.5/1 会欠引导）
2. **画布**：480×640 脸部 token 太少 → `--size 640x864 --frames 65`
3. **帧数**：越长漂越多 → 65/49
4. **LoRA strength**：`--lora-strength 0.8` 贴近基础模型轨迹

A/B 方法：固定 `--seed` 同图同提示词只改一个变量（实物对：bernini_ab_s4.mp4 vs bernini_ab_s8.mp4，seed=12345）。

## 7. 故障速查

| 现象 | 原因 / 解决 |
|------|------|
| `workflow_unknown_nodes: BerniniStudio` | 节点包没装/没重启 → 第 1 节装 2 个包并重启 |
| `shape_mismatch: expected INT, got str '5'`（节点 31） | 前端 Int 存的字符串 → 映射 PrimitiveInt 时值必须改整数 `[5]` |
| 提交即 `ws_disconnected`，boot log 停在 `Requested to load WAN21`，进程消失 | **撞 cgroup 128G 被 SIGKILL**：上一跑（如 SCAIL 922 帧）的模型/帧缓存没释放，基线 ~130G+，再装 14.5G 模型即死 → **重启 ComfyUI 清缓存再跑**；`cat /sys/fs/cgroup/memory.current` 看基线 |
| BerniniStudio 卡住/报连接错 | `auto_enhance=true` 在调 Ollama → widgets[9] 改 false |
| `unknown_enum_value` at 7/10（LoRA） | 作者 json 的 LoRA 路径是 Windows 反斜杠子路径，Linux 不认 → 裸文件名放 loras/ 根目录；未提供前 mode=4 bypass |
| `Requested to load WAN21` 后 OOM（显存） | 双 int8 14.5G 轮换 + 大画布超 24G → 降 34/35 画布（640×864 实测峰值 14.6G） |
| 挂 LoRA 后 SamplerCustom OOM（`int8_linear→torch.cat`） | 高噪模型 + 3.7G bf16 LoRA 顶住 24G → ① `--lowvram` 重启 ② 像素×帧预算 ≤ ~36M（65f@640×864 或 81f@512×688，见 5.5 节） |
| **上次 OOM 过的配置这次却 OOM**（甚至 n=1 都挂） | OOM 后进程内显存碎片化，后续跑啥都炸 → **OOM 事件后必须重启 ComfyUI 再跑**（2026-07-25 实测：同进程连跑 6 发后 17 帧都 OOM，重启即恢复） |
| scp 拉产物报 No such file（路径含 `%date:...%`） | SFTP 模式远端单引号不被解释 → 服务器先 `cp` 到 /tmp 简单名再拉（run_bernini.py 已内置） |
| int8_convrot 加载报不支持 | ComfyUI 太旧 → 用 master（boot 应见 `Native ops: ... convrot_w4a4`） |
| 只有首帧 png 没有 mp4 / 快速"假成功" | 参考 SCAIL 第 9 节：节点校验失败被静默跳过 → 查 boot log `Failed to validate prompt` |

> 网络铁律同 SCAIL：github 走 ghfast tar、模型优先 ModelScope（hf-mirror 会按 IP 限速）、后台下载用 setsid、pkill 用 `[x]` 括号技巧防自杀。
