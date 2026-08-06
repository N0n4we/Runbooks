# MiniMax H3 全模态视频生成 — 部署与执行 Runbook

> **两台目标机，选型不同且都不是偏好问题——是内核可用性问题。见 §12（现行主机）。**
>
> | 档位 | 主机 | GPU | DiT / 编码器 | 后端 | 实测 |
> |---|---|---|---|---|---|
> | `--host 5090`（缺省） | `PER3cU@wp08:21054` | RTX 5090 32G **sm_120** | int8 convrot / int8 25.28G | **triton**（需 triton ≥3.3，见 §12-G） | **4.09 s/it**，四模态+LoRA 全过（§12） |
> | `--host 4090` | `kIYRa5@wp08:25304` | RTX 4090 24G sm_89 | int4 convrot / int4 13.93G | eager | ~5.75 s/it，未上机实渲染 |
>
> 底图：四个 Qwen3VL 模板（T2VA / I2VA / FL2VA / R2VA），推理链封装在同一子图（实例 #105）。
> 一键脚本：`run-h3.py`（文本 / 首帧 / 首尾帧 / 参考图；支持 `--lora`；参考视频与音频不支持，见 §11）。
>
> ⚠️ 阅读顺序：**§12 是现行主机（5090 + int8）的完整实测结论，与它冲突的一律以 §12 为准。**
> §9 / §10 / §11 是四模板通用结论。§1 的环境记录属 4090，§2 的 int8 选型表与 §5 的验证记录
> 是**更早的 3090 机器**历史记录，保留作对照。

## 0. 这是什么

MiniMax H3 = 全模态打包 DiT：文本/图像/视频/音频统一上下文理解 → 输出**带原生双声道音轨**的视频（人声/音效/配乐单次前向一起出，非事后叠加），最高 2K、24fps、约 15s。ComfyUI ≥0.30.0 原生支持（`comfy_extras/nodes_minimax_h3.py` 四节点）。

> ⚠️ 旧版本这里写「**零第三方节点包**即可跑」——对**四个 Qwen3VL 新模板已失效**：它们引用
> VideoHelperSuite / KJNodes / Easy-Use（可用）以及 Upscaler-Tensorrt / Rife-Tensorrt /
> AILab QwenVL（本机不可用，脚本已自动旁路）。详见 §10。

- 两个底模家族：**fl2va**（T2V 纯文本 + I2V 首/末帧）与 **ref2va**（参考驱动：图/视频/音频混合条件），权重不通用。
- 核心节点 `MiniMaxH3ImageToVideo`（t2va/fl2va：prompt + 可选 first/last_frame）与 `MiniMaxH3ReferenceToVideo`（ref2va：动态槽 ref_image_0..8 / ref_video_0..2 / ref_video_audio_0..2 / ref_audio_0..2）。**注意**：节点本身有这些槽，但 R2VA-Qwen3VL 模板的子图边界只暴露了 ref_image_0/1 → 实际只能用 2 张参考图，见 §11。
- 无负面词、无 CFG（BasicGuider=1）；采样 res_multistep + simple 20 步为官方模板默认。
- 帧数网格 **17k+5 @24fps**（5s=124 帧），训练范围 124~362 帧（≈5~15s）；画布 32 倍数，原生 768 短边、面积上限 768×1344。

## 1. 环境与网络（`--host 4090` 档位，wp08:25304 / RTX 4090 实测，2026-08-06；5090 见 §12）

- **本机 GitHub / HuggingFace / hf-mirror 全不可达**，只有 **ModelScope + aliyun pypi** 通；ComfyUI 主程序走 `gh-proxy.com` 中转（旧版这里写「codeload.github.com 直连可达」——那是旧机器，本机实测不通）：
  ```bash
  mkdir -p ~/comfy && cd ~/comfy
  curl -sL -o ComfyUI.tar.gz "https://gh-proxy.com/https://codeload.github.com/comfyanonymous/ComfyUI/tar.gz/refs/heads/master"
  gzip -t ComfyUI.tar.gz && tar xzf ComfyUI.tar.gz && mv ComfyUI-master ComfyUI
  pip install -r ComfyUI/requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --retries 30 --timeout 60
  pip install comfy-cli modelscope -i https://mirrors.aliyun.com/pypi/simple/
  ```
- 裸容器无 curl/ffmpeg/git：`sudo apt-get install -y curl wget ffmpeg libgl1 unzip git`（apt 源已是阿里；后台 apt 被 kill 后 dpkg 锁会自动释放，直接重跑即可）。**git 必装**：`comfy run` 的 GitPython 无 git 直接拒绝执行。
- 模型走 ModelScope CLI（~10MB/s、自动续传）：`modelscope download --model Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot "<repo路径>" --local_dir ~/comfy/ComfyUI/models`。**下载后必须剥 L2P 尾标记**，见 §8。
- torch 2.7.0.dev20250310+cu128（镜像预装，<2.8 → legacy ModelPatcher）。**cu128 < cu130 → comfy-kitchen 的 cuda 后端被禁**，triton 后端有 `int8_linear` 但**无 `convrot_w4a4_linear`** → 纯 int4 只能 eager（慢），int8 层可吃 triton 加速，故 `--dit mixed` 在本机是质量/速度/显存的折中选项（详见 §7-B）。
- 启动（**去掉 `--lowvram`**，让 int4 编码器 13.93G 驻留 GPU，理由与实测数据见 §9；**绑回环**，避免把无鉴权 API 暴露到所有网卡，`comfy run` 走 127.0.0.1 即可）：
  ```bash
  setsid nohup python3 main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch \
    > /tmp/comfy_boot.log 2>&1 </dev/null & disown
  ```
- 验证：`/object_info` 应有 MiniMaxH3ImageToVideo / MiniMaxH3ReferenceToVideo / MiniMaxH3SigmaShift / EmptyMiniMaxH3LatentAV；装齐第三方包后实测 1319 类（§10）。

## 2. 模型文件（24G 选型）— ⚠️ 旧方案（int8 / 3090），现行见 §7-B

> 保留作对照。**现行是 int4 convrot**（§7-B），文件名与大小都不同，不要照本节抄。

| 目标路径 | 大小 | 说明 |
|---|---:|---|
| `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 20.97G | T2V/I2V 底模（官方模板默认精度） |
| `diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors` | 20.97G | R2V 底模 |
| `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | 27.14G | 旧机（3090 sm_86）选 int8：模板默认的 nvfp4_awq 是 Blackwell(sm_120) 专用；bf16 51.5G 无必要 |
| `vae/minimax_h3_video_vae_fp16.safetensors` | 5.21G | 视频 VAE |
| `vae/minimax_h3_audio_vae_fp32.safetensors` | 0.61G | 音频 VAE（原生音轨） |

合计 ~75G。CLIPLoader 的 type 必须保持 `minimax`（这一条对 int4 同样成立）。

## 3. 工作流结构（四模板，Qwen3VL 版）

现行底图是仓库内四个模板，每个模态一枚（旧的 `H3文生视频工作流.json` / `H3参考生视频工作流.json` 已删除）：

| 模态 | 模板文件 | 根图节点数 | 触发方式 |
|---|---|---:|---|
| T2V  | `MiniMaxH3-T2VA-Qwen3VL.json`  | 13 | 纯 `--prompt` |
| I2V  | `MiniMaxH3-I2VA-Qwen3VL.json`  | 14 | `--image`（根图 LoadImage #114） |
| FL2VA| `MiniMaxH3-FL2VA-Qwen3VL.json` | 15 | `--last-image`（#114 首 / #147 尾） |
| R2VA | `MiniMaxH3-R2VA-Qwen3VL.json`  | 15 | `--ref-image`（#114 / #152） |

**关键结构（四者一致）**：真正的推理链**全在同一个子图**里——子图定义 `definitions.subgraphs[0]`（id `4c314f31…`，18 个内部节点），根图只有子图实例 **#105** + TensorRT 上采样/RIFE 插帧后处理链 + VHS_VideoCombine + SaveVideo #92 + ResolutionSelector #115 + easy showAnything + 若干 MarkdownNote。

```
根图:   ResolutionSelector(115) → 105.width/height；LoadImage(114/147/152) → 105 参考/帧输入
        105 VIDEO ──link194──→ SaveVideo(92)          ← 脚本认这条出片
        105 → Upscaler/Rife TensorRT 链(128,127,146,145) → VHS_VideoCombine(126)  ← 已旁路(§10)
子图内: UNETLoader 6 / CLIPLoader 13(type=minimax) / VAELoader 11(视频) 24(音频) /
        RandomNoise 15 / KSamplerSelect 17 / BasicScheduler 9 /
        PrimitiveFloat 111(秒) → ComfyMathExpression 107(17k+5 吸附) →
        MiniMaxH3ImageToVideo 104（R2VA 为 MiniMaxH3ReferenceToVideo 149）→
        BasicGuider 16 → SamplerCustomAdvanced 14 → VAEDecode 10 / VAEDecodeAudio 23 →
        CreateVideo 91 → 子图 VIDEO 输出
        AILab_QwenVL 增强器(150/151/154) 与 PathchSageAttentionKJ(147/148) ← 已旁路(§10)
```

**权重名双写**：子图实例 #105 的 promoted widgets `[5..8]`（unet/clip/vae_video/vae_audio）是权威值，子图内部 loader #6/#13/#11/#24 各自还有一份，`properties.models` 里第三份。脚本 `rewrite_weights()` 三处同步改写。模板原值是 **NVFP4（Blackwell 专用，4090 不能用）**，必须换掉。

**已修的模板 bug**：R2VA 的 promoted 值是 Ref2VA，但内部 `UNETLoader#6` 残留着 FL2VA 的默认值（会加载错模型）；`properties.models[*].url` 是失效的 HF 链接且带旧 NVFP4 basename，与新 name 自相矛盾 → 一并删除。

**新增子图输入**（旧模板没有）：`model_name` / `preset_prompt` / `keep_last_prompt` / `steps`，对应 promoted widgets `[9..12]`。

- 105.widgets_values = [prompt, width, height, 秒, seed, unet, clip, 视频vae, 音频vae, model_name, preset_prompt, keep_last_prompt, steps]（instance 代理值优先于内部 widgets，**补丁要双写**）。
- links 格式陷阱：顶层 `links` 与子图 `definitions.subgraphs[].links` 两种格式（列表 / dict）都要兼容——脚本 `Graph._l()` 做了归一化。
- 子图边界（`subgraphs[0].inputs`）本身就是接线目标：`origin_id = -10` 表示"来自子图输入边界"，`target_id = -20` 表示"去往子图输出边界"。做连通性检查时必须把这两个虚拟节点算进合法 id，否则会误报大量悬挂边。
- **R2V 参考槽只暴露了 2 个**：核心节点 `MiniMaxH3ReferenceToVideo#149` 有 ref_image_0/1/2 + ref_video_0 + ref_video_audio_0 + ref_audio_0，但子图边界只有 `ref_images.ref_image_0` / `ref_image_1`，根图只有 LoadImage #114/#152 → 实际能力 = 2 张参考图。详见 §11。
- `#149 widgets = [prompt(被接管), width, height, length, ref_image_size]`；ref_image_size: `match`=参考图缩到画布面积（快）/ `max`=2048 短边（身份保真，慢数倍）。

**SigmaShift（可选，四模板通用）**：`MiniMaxH3SigmaShift`（shift_video 默认 12 / shift_audio 默认 3）不在模板里，节点内建值生效。脚本的 `--shift-video/--shift-audio` 会在 UNETLoader 后动态插入该节点并**接管 UNET 的全部 MODEL 出边**（BasicGuider 与 BasicScheduler 必须共用同一份 patched model，否则调度表与 DiT 内部 video/audio shift 不一致）。`--lora` 用同一手法，两者叠加时链路为 `UNET → LoRA → SigmaShift → 引导器/调度器`。

## 4. 一键脚本 run-h3.py

```bash
# T2V
python3 run-h3.py --prompt "……"
# I2V（首帧）/ FL2VA（首尾帧）
python3 run-h3.py --prompt "……" --image first.png [--last-image last.png]
# R2V（参考图，≤2；--ref-video/--ref-audio 当前不支持，传入即报错，见 §11）
python3 run-h3.py --prompt "让 <Picture 1> 的角色走进 <Picture 2> 的场景……" \
    --ref-image role.png --ref-image scene.png
# 结构自检（不连服务器、不渲染，四模态都能跑）：
python3 run-h3.py --dry-run --prompt "……" [--image a.png | --ref-image r.png]
# 常用：--seconds 5（17k+5 吸附）/ --frames 124 / --size 768x1344（32 倍数）
#       --aspect "16:9 (Widescreen)" --megapixels 0.4（原生画质 1.0≈1344x768）
#       --steps 20 / --sampler res_multistep / --scheduler simple / --seed N
#       --shift-video 12 --shift-audio 3（SigmaShift，缺省不插节点）
#       --lora NAME[:strength]（LoraLoaderModelOnly，缺省 strength 1.0）
#       --host 5090|4090（目标机档位，缺省 5090；同时决定 --dit/--clip 缺省与 SSH 目标）
#       --dit int8|fp8|int4|mixed（DiT 量化档位；两来源命名不同，见 §12-D）
#       --clip int8|int4（文本编码器档位；int8 25.28G 需 32G 卡，24G 上会告警）
#       --enhance / --postprocess / --sage（恢复默认被旁路的链路，见 §10）
#       --ref-image-size match|max / --out x.mp4 / --no-download
```

行为：模态互斥校验 → 上传输入（run_id 前缀）→ 内存补丁共享 json（**NVFP4→int4 convrot 权重名**、模型名、种子、采样、时长、画布、参数化、旁路降级、可选 LoRA/SigmaShift）→ `comfy run` 提交 → run_id 隔离轮询 → 拉回 ffprobe 验证（视频帧数 + **音轨存在性**）。输出 h264+aac mp4（QuickTime 可播）。

**`--dry-run` 是本地唯一的验收手段**：完全不碰 SSH，把补丁后的图 dump 到 `/tmp/run_h3_dryrun_<mode>.json`，并自断言「核心 SaveVideo 链完整 ∧ 无 HF 依赖 ∧ 无 TensorRT 依赖」。服务器不在时靠它回归。

## 5. 验证记录 — ⚠️ 旧机器（wp08:33307 / RTX 3090 24G / int8 / `--lowvram --reserve-vram 5`，2026-08-04）

> **本节是历史记录，配置已不适用于现行 4090 + int4 + NORMAL_VRAM 方案**（见 §9）。
> 保留是因为其中的 OOM 归因与帧数/音轨门禁方法仍有参考价值。
> 四个 Qwen3VL 新模板在 4090 上的真实渲染记录**尚未补齐**（服务器停机），待补。

**`--reserve-vram 5` 的由来（关键教训，旧机器 int8 场景）**：仅 `--lowvram` 时，legacy ModelPatcher（torch 2.7 <2.8 无 DynamicVRAM）把 21G int8 DiT **整载**进 24G VRAM（`loaded completely 19996 MB`），剩 ~4G 不够 int8_linear 反量化 `torch.cat` 暂存 + 60k token 注意力激活 → 采样器 OOM。加 `--reserve-vram 5` 后 DiT 变部分加载（14.8G 驻留 / 5.2G offload 流式），四发全过。

| 项 | 配置 | 结果 | 墙钟 |
|---|---|---|---|
| T-T2V | 864×480×124f（模板默认 16:9 0.4MP），纯文本 | ✅ h264+aac 双声道，5.167s | ~16min（含首载） |
| T-I2V | 576×736×124f（3:4 0.4MP），首帧=test_girl.png | ✅ 同上 | ~18min |
| T-R2V | 864×480×124f，参考图×1 + 参考视频×1（带音轨）+ 独立音频×1 全模态混合 | ✅ 同上 | ~31min |
| T-grid | 768×768×56f（`--frames 56 --size 768x768 --seed 42`） | ✅ 精确 56 帧 2.33s | ~11min |
| T-merge+shift | 合并模板 i2v（文生模板动态接首帧）+ SigmaShift（v8/a2），576×736×56f | ✅ 2.33s 双声道 | ~11min |

- 帧 sanity：YAVG 帧间有变化（非黑帧/定格）；volumedetect mean -16~-43dB（按需出声）。
- 加载行为：qwen int8 27G 全程 CPU fp16（lowvram 策略）；DiT 部分驻留 + 流式；VAE 解码时另载 ~5G。
- 墙钟含 qwen CPU 编码（占大头）；R2V 另加参考编码。产物均在 `~/Downloads/h3_*_first.mp4`、`h3_t2v_square.mp4`。

## 6. 故障速查

| 症状 | 原因/处理 |
|---|---|
| SamplerCustomAdvanced OOM，栈落 `int8_linear→torch.cat`，boot log 见 DiT `loaded completely ~20G` | **旧机器 int8 场景**：legacy ModelPatcher 把 21G DiT 整载进 24G，无暂存余量 → 启动加 `--reserve-vram 5` 强制部分加载流式（§5 历史记录）。现行 int4 DiT 仅 10.56G，不适用 |
| `comfy run` 提交即败，报 git/GitPython | 裸容器无 git → `apt-get install git` |
| 官方 i2v 模板校验 `required input 'image' is missing`（节点 119） | 旧模板遗留问题，已随旧模板删除；四个 Qwen3VL 模板各自有独立 LoadImage，不再适用 |
| CLIPLoader 报 nvfp4 相关错 | 模板默认 qwen3vl nvfp4 是 Blackwell(sm_120) 专用 → 换 `qwen3vl_32b_minimax_h3_int4_convrot`（脚本 `rewrite_weights()` 已自动，type 保持 `minimax`） |
| UNETLoader 加载了错的 DiT（R2V 出图像是 FL2VA 行为） | R2VA 模板内部 `UNETLoader#6` 残留 FL2VA 默认值，与 promoted 值不一致 → 脚本已修；手改模板时记得三处（promoted #105[5] / 内部 #6 / properties.models）同步 |
| `--ref-video` / `--ref-audio` 报错退出 | 预期行为，非 bug：R2VA 子图未暴露对应边界，见 §11 |
| 产物没有音轨 / 出片是上采样后的版本 | 模板默认旁路了 SaveVideo#92 而走 TensorRT+VHS 链；脚本反过来强制激活 #92 并旁路 TRT 链（§10）。若手动跑模板需自行调整 |
| `--upscale` 报 `BuilderFlag has no attribute 'FP16'`（node 128） | 服务器装的是 **TRT 11.2.1.2**，TRT 11 删除了弱类型精度开关 `BuilderFlag.FP16`；老节点无条件引用 → AttributeError。已给节点 `trt_utilities.py` 第 217 行打 `hasattr` 守卫补丁 → TRT 11 上退化为 **fp32 构建引擎**（§12-H） |
| `--upscale` 报 `'NoneType' object has no attribute 'set_input_shape'`（node 127） | 引擎构建/加载成功但 `create_execution_context()` 需预分配 **~17.6GB** 显存，而扩散模型仍驻留（~26GB）→ 32G 卡 OOM → 返回 None。已给节点 `nodes/upscaler_tensorrt.py` 打补丁：分配前先 `mm.unload_all_models()` 腾显存（§12-H） |
| 想降级 tensorrt 到 10.x 保 fp16 | **被 GFW 堵死**：PyPI 上 `tensorrt_cu12_libs` 只有 sdist(永不发布 wheel) 且其 `wheel-stub` 后端从 `pypi.nvidia.com` 拉库，而该站从本机访问 `tensorrt_cu12_libs/bindings` 全 404、一体化 `tensorrt` 也只有 sdist → 无任何可达的 10.x 预编译轮子。fp32 已够用，勿再尝试（§12-H） |
| 校验 `class_type 'X' not found` | ComfyUI <0.30.0 或未重启 → 升级 master 并重启；object_info 必须解析 body |
| 换精度/配置连跑 OOM | 进程内显存碎片化 → 重启 ComfyUI 再跑（老经验） |
| ssh 执行 pkill 后断（exit 255） | kill 与启动分两条 ssh（pkill -f 自匹配） |
| pgrep 误判进程在/不在 | 模式串会被远端 bash -c 自身命令行匹配 → 方括号打断 + **检查与启动分两条 ssh**（同串含明文目标串时括号也救不了） |
| 轮询永不退出 | pgrep 模式用方括号打断自匹配（`comfy r[u]n`、`[m]ain.py`） |
| 拉回 mp4 打不开 | 等 comfy run 进程退出再拉（moov 最后落盘），本地 ffprobe 门禁 |
| 双进程下同模型下载互踩 .incomplete | modelscope CLI 无文件锁；并行下载同文件会互相破坏 → 一个文件一个下载进程，核对清单再并行 |

## 7. 两种精度方案模型清单

### A. INT8

来源：Comfy-Org/MiniMax-H3（ModelScope，本机 ~12MB/s）

| 组件 | 文件 | 大小 |
|---|---|---|
| DiT FL2VA | diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors | 20.97G |
| DiT Ref2VA | diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors | 20.97G |
| 文本编码器 | text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors | 27.14G |
| 视频 VAE | vae/minimax_h3_video_vae_fp16.safetensors | 5.21G |
| 音频 VAE | vae/minimax_h3_audio_vae_fp32.safetensors | 0.61G |

### B. INT4

来源：Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot（HF，本机 HF 不可达需镜像/中转）

| 组件 | 文件 | 大小 |
|---|---|---|
| DiT FL2VA | MiniMax_H3_FL2VA_pruned_int4_convrot.safetensors | ~11.3G |
| DiT Ref2VA | MiniMax_H3_Ref2VA_pruned_int4_convrot.safetensors | ~11.3G |
| 文本编码器 | qwen3vl_32b_minimax_h3_int4_convrot.safetensors | 15.0G |
| 视频 VAE | minimax_h3_video_vae_fp16.safetensors | 5.21G |
| 音频 VAE | minimax_h3_audio_vae_fp32.safetensors | 0.61G |

## 8. modelscope 下载的 L2P 标记坑（必读）

modelscope CLI 下到的 .safetensors 会在**文件尾部附加 `L2P_bypass_<path>` 标记字节**（反检测页脚），safetensors 严格模式因此报 `SafetensorError: incomplete metadata, file not fully covered`，ComfyUI 无法加载。文件尺寸看起来对、常规 json 解析也能过，但 safetensors 库拒读。

**修复**：按 header 声明长度截断文件，去掉尾部标记。python 一步：
```python
import struct, json, os
p="path.safetensors"; f=open(p,"rb"); hlen=struct.unpack("<Q", f.read(8))[0]
header=json.loads(f.read(hlen)); keys=[k for k in header if not k.startswith("__")]
decl=8+hlen+max(header[k]["data_offsets"][1] for k in keys)
os.truncate(p, decl)   # 去掉 L2P_bypass 尾部
```
凡 modelscope 下的 safetensors（尤其 Abiray INT4 套件）下载后都要先剥标记再验载（`safetensors.torch.load_file` 一次）才能用。Abiray INT4 全套已按此修复（FL2VA/编码器）。

## 9. GPU 利用率低之谜：编码器跑在 CPU 上（已验证，可消除）

**现象**：H3 出片慢、nvidia-smi 显示 GPU 利用率长期近 0、功耗 ~32W（空转）。

**根因（实测确认）**：
1. `--lowvram` 启动参数会强制把 text encoder（qwen int4）offload 到 **CPU**。明明 int4 编码器只有 14.26GB、整块装得进 24G 显存（日志 `full load: True`），仍被压到 CPU。一次 T2V ~8.3min 里约 6min 全耗在 CPU 编码，期间 GPU 完全空转。
2. 次因：`comfy/quant_ops.py` 的后端禁用逻辑——torch cu128 < cu130 时 `ck.registry.disable("cuda")`（日志 WARNING "You need pytorch with cu130 or higher"）；NVIDIA 路径不传 `--enable-triton-backend` 就默认 `disable("triton")`。结果只剩 eager 后端；而 FL2VA/Ref2VA 的 **convrot_w4a4 算子只有 cuda/eager 实现（triton 没有）**，cuda 被禁后只能走 eager 纯 torch 反量化 → 采样 20 步 ~5.75s/it（对 4090 偏慢）。此项需升 torch≥cu130 或加 `--enable-triton-backend` 才解，优先级低于第 1 条。

**消除办法（已验证）**：重启时去掉 `--lowvram`，用 NORMAL_VRAM：
```
python3 main.py --port 8188 --reserve-vram 4
```
日志显示 `Set vram state to: NORMAL_VRAM`、`CLIP/text encoder model load device: cuda:0`、`loaded completely; 14258.82 MB loaded, full load: True`。

**验证数据（单次文字编码对比）**：--lowvram(CPU) ~380s ? 去掉 lowvram(GPU) **9.06s**，约 40 倍提速。

**注意点**：编码器 14.7G 驻留后，DiT int4(11.3G) 同时放不下（~26G > 24G）。靠"先 GPU 快速编码 → 逐出 → 再载 DiT 采样"的串行换载即可；全跑墙钟编码段从 ~6min 压到 ~10s。

## 10. 旁路/降级决策（task 7，四个 Qwen3VL 模板通用）

四个新模板（`MiniMaxH3-{T2VA,I2VA,FL2VA,R2VA}-Qwen3VL.json`）比旧模板多挂三条本机不可用/未必可用的链路。`run-h3.py` 在 `apply_degradation()` 里用 **ComfyUI 原生 bypass（节点 `mode=4`）** 统一降级，缺省全开降级，显式 `--enhance`/`--postprocess`/`--sage` 恢复。选 bypass 而非剪线重连的原因：子图 I/O 是虚拟边界节点（`-10`/`-20`，不在 `nodes` 里，边靠 `definitions.subgraphs[].inputs/outputs` 的 `linkIds` 记账），跨界重连易写坏边界记账；bypass 是图→prompt 转换期的原生语义——被旁路节点不执行，其输出按类型就近透传到同类型输入，天然等价于"无此节点"的核心链。

| 链路 | 现状（task 6 实测） | 决策（缺省） | 恢复开关 | 落地方式 |
|---|---|---|---|---|
| **(a) AILab QwenVL 增强链** | 依赖 `Qwen3-VL-8B-Heretic-Stable` / `Qwen3.5-9B-Unredacted-MAX`，**HF 不可达**，模型拉不下来 | **旁路**：增强器节点 `mode=4`，其 STRING 输出按类型透传其 STRING 输入（子图边界原始 prompt）→ 原始 prompt 直达主推理节点 prompt 口 + 子图 output（easy showAnything 预览），不触发 HF 下载 | `--enhance` | 增强器 id 随模板：t2v=#151(`AILab_QwenVL_PromptEnhancer`)，i2v=#150，fl2va=#151，r2v=#154（均 `AILab_QwenVL_Advanced`） |
| **(b) TensorRT 上采样 + RIFE 插帧后处理** | ⚠️ **2026-08-08 起 `--upscale` 已可跑通**（见 §12-H）：TRT 链的**超分两段 `#128/#127` 现可用**（`Upscaler-Tensorrt` 已按 §12-H 打过两个节点补丁）。RIFE **插帧 `#146/#145` 仍不可用**：模板引用 `AutoRifeTensorrt / AutoLoadRifeTensorrtModel`，而已装的 `ComfyUI-Rife-Tensorrt` 只提供 `RifeTensorrt / LoadRifeTensorrtModel`，名字不匹配，激活即被服务端拒。**缺省仍旁路**整条 `#128/#127/#146/#145/#126`(VHS)。⚠️ **模板默认把 `SaveVideo #92` 置 `mode=4`、以 `VHS_VideoCombine #126` 为出片**，而 VHS 的 IMAGE 输入来自 TRT 上采样链——缺 engine 即断链且无产物。故 `apply_degradation` 缺省**激活 `SaveVideo #92`（`mode=0`）**并旁路 TRT 链：核心出片经 link194 直连 #92 → 稳出 h264+aac（脚本以 `H3_<run_id>` 前缀认 #92 产物）；`--upscale` 恢复 `#128/#127/#126`，此时 #92 仍并行激活（保住核心产物），超分产物以 `H3up_<run_id>` 前缀经 VHS#126 出片 | `--upscale`（TRT 超分，仅超分；`--postprocess` 同理恢复 TRT 链） | 引擎首次自动 build（~43s，缓存于 `models/tensorrt/upscaler/*.trt`）；出片两路并存：`H3_<id>.mp4`（原始）+ `H3up_<id>.mp4`（2x 超分） |
| **(c) KJNodes SageAttention** | `PathchSageAttentionKJ`（KJNodes 已注册），但 SageAttention 内核在本机 cu128 上**未必可编** | **旁路**：`mode=4`，MODEL 输出按类型透传 MODEL 输入 → 回退默认注意力（内核缺失也能跑）。同链 `ModelPatchTorchSettings` 保留（仅 torch 后端开关，无第三方内核依赖） | `--sage` | Sage 节点 id：t2v/i2v/r2v=#147，fl2va=#148 |

**核心链定义（旁路后必须完好、全 `mode=0`）**：子图内 `UNETLoader#6 →(Sage 旁路透传)→ BasicGuider#16 / BasicScheduler#9 → SamplerCustomAdvanced#14 → VAEDecode#10 + VAEDecodeAudio#23 → CreateVideo#91 →` 子图 VIDEO 输出 `→ SaveVideo#92`。`--dry-run` 会 dump 补丁后工作流并断言：核心链完整 ∧ 无 HF 依赖 ∧ 无 TRT 依赖（缺省档），四模态已验证通过。

**与 SigmaShift/LoRA 的叠加**：`--shift-*`(task5)/`--lora`(task8) 均在 `UNETLoader #6` 后接管其全部 MODEL 出边（含 →BasicScheduler 与 →Sage）。Sage 旁路（透传）与之正交：链路变为 `UNET→[LoRA]→[SigmaShift]→(Sage 透传)→ModelPatchTorchSettings→BasicGuider`，MODEL 边一致。

## 11. R2V 参考输入的真实能力（2026-08-07 核实）

**结论：当前只支持 2 张参考图；`--ref-video` / `--ref-audio` 不支持，传入即报错退出。**

核心节点 `MiniMaxH3ReferenceToVideo#149` 自身有六个参考槽：

```
inputs[3] ref_images.ref_image_0        ← 已接线（子图边界 13）
inputs[4] ref_images.ref_image_1        ← 已接线（子图边界 14）
inputs[5] ref_images.ref_image_2        ← link=None，边界未暴露
inputs[6] ref_videos.ref_video_0        ← link=None，边界未暴露
inputs[7] ref_video_audios.ref_video_audio_0  ← link=None，边界未暴露
inputs[8] ref_audios.ref_audio_0        ← link=None，边界未暴露
```

而 `MiniMaxH3-R2VA-Qwen3VL.json` 的 `subgraphs[0].inputs` 只到 `ref_images.ref_image_1`，
根图也只有两个 `LoadImage`（#114 / #152）。**节点有槽 ≠ 模板接了线。**

### 为什么要报错而不是忽略

修复前：`main()` 的断言放行 ≤9 图 / ≤3 视频 / ≤3 音频，`upload_inputs()` 把它们**全部 scp
上传**成 `refvid*` / `refaud*`，但没有任何代码把它们接进节点。结果是**静默产出一个完全忽略
了参考视频/音频的视频，不报错、不警告**——用户以为参考生效了，实际没有。这是最坏的失败方式，
所以现在改成入口处直接 `sys.exit(1)` 并说明原因。第 3 张起的参考图同理拒绝。

### 结论：参考能力就对齐工作流，不扩展（2026-08-08 定案）

`run-h3.py` 的参考能力**按 `MiniMaxH3-R2VA-Qwen3VL.json` 实际接线为准**：`--ref-image` 上限
2 张，`--ref-video` / `--ref-audio` 传入即 `rc=1` 退出。这是**模板的设计能力，不是待修复的缺口**，
脚本不做超出模板的事。

要突破得改模板本身（在 `subgraphs[0].inputs` 追加边界、子图内补 link 接到 #149 的 inputs[5..8]、
根图补 `LoadImage` / `LoadVideo`→`GetVideoComponents` / `LoadAudio`），且**ComfyUI 是否接受运行时
动态新增的 subgraph 边界从未验证**。这属于改工作流，不在本脚本职责内，**已明确不做**。

若将来真要做，参考视频的额外约束（节点源码）：≥5 帧、会吸附 17k+5、截到不长于成片帧数，
Qwen 以 2fps 带时间戳读取。

---

## 12. 5090 / INT8 部署与实测（2026-08-08，**现行主机**）

`PER3cU@wp08.unicorn.org.cn -p 21054` / RTX 5090 32G **sm_120** / ComfyUI **0.31.0** /
torch `2.7.0.dev20250310+cu128`（cu128，勿动，见 §12-G）/ **triton 3.3.1（用户级，必装）** /
comfy-kitchen 0.2.28 / 驱动 570.86.10。

### 12-0. 从零部署这台机器的步骤（**按序执行，每步都有验证门禁**）

> triton 3.3.1 是**部署的必需步骤**，不是可选优化：不装它就不能加 `--enable-triton-backend`，
> 而不加就只有 eager（慢一倍）；装了却不加同样浪费。两者必须成对。

```bash
# 1. 前置：确认 GPU 与驱动（驱动决定 cu 版本上限，不可改）
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
#    期望 NVIDIA GeForce RTX 5090 / 570.86.10 / 32607 MiB。驱动 <580.65 → 只能 cu128（§12-G）

# 2. ComfyUI 0.31.0 + 依赖（pip 源用 aliyun：它是唯一收齐全部钉版包的镜像，见 §12-D 注）
cd ~/comfy/ComfyUI
python3 -c "import torch;print('torch==%s' % torch.__version__)" > /tmp/torch_pin.txt   # 钉住 torch
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ \
  -c /tmp/torch_pin.txt --retries 10 --timeout 60
python3 -c "import torch;print(torch.__version__, torch.version.cuda)"   # 门禁：必须仍是 cu128

# 3. 【必需】用户级升 triton 到 3.3.1 —— 系统自带的 3.2.0 在 sm_120 上编译即崩
pip install --user --no-deps triton==3.3.1 -i https://mirrors.aliyun.com/pypi/simple/
python3 -c "import triton,os;print(triton.__version__, os.path.dirname(triton.__file__))"
#    门禁：必须输出 3.3.1 且路径在 ~/.local/...（不是 /usr/local/...）

# 4. 门禁：确认 triton 真能为 sm_120 编译（不通过就别往下走，否则 ComfyUI 会 SIGABRT）
#    注意必须**写成文件再执行**：triton.jit 要 inspect.getsourcelines() 取源码，
#    用 `python3 - <<EOF` 从 stdin 喂会 OSError: could not get source code（与 triton 无关）
cat > /tmp/triton_probe.py <<'EOF'
import torch, triton, triton.language as tl
@triton.jit
def k(x, o, N: tl.constexpr):
    v = tl.load(x + tl.arange(0, N)); tl.store(o, tl.sum(v, axis=0))
x = torch.randn(256, device="cuda"); o = torch.empty((), device="cuda")
k[(1,)](x, o, N=256); torch.cuda.synchronize()
assert abs(o.item() - x.sum().item()) < 1e-2
print("triton sm_120 编译 OK")
EOF
python3 /tmp/triton_probe.py
#    triton 3.2.0 下这里会打印 'sm_120' is not a recognized processor 并
#    LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.shfl.sync.bfly.i32 然后 Aborted

# 5. 自定义节点（VHS / KJNodes / Easy-Use；两个 *-Tensorrt 包会 IMPORT FAILED，属预期，
#    因为默认旁路 TRT 链、未装 tensorrt）
# 6. 权重：按 §12-D 的五个文件下载到 models/ 下（约 69.75 GiB，官方 Comfy-Org 仓库）
# 7. 启动（见 §12-A，必须带 --enable-triton-backend）
# 8. 验收：run-h3.py 跑一发，ffprobe + **抽帧看画面**（§12-F 的教训：ffprobe 过了不等于内容对）
```

重建机器时第 3 步最容易漏——它是用户级安装，不在 `requirements.txt` 里，`pip freeze` 也容易被
系统的 3.2.0 混淆。**判断当前是否满足：`python3 -c "import triton;print(triton.__version__)"`。**

### 12-A. 启动命令（**要先满足 §12-G 的 triton ≥3.3 前置条件**）

```bash
cd ~/comfy/ComfyUI && setsid nohup python3 main.py \
  --listen 127.0.0.1 --port 8188 --disable-auto-launch \
  --enable-triton-backend \
  > /tmp/comfy.log 2>&1 & disown
```

⚠️ **`--enable-triton-backend` 只有在 triton ≥3.3 时才可加。** 系统自带的 triton **3.2.0**
带的 LLVM 不认识 sm_120，加了会让进程**硬崩**：

```
'sm_120' is not a recognized processor for this target (ignoring processor)
LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.shfl.sync.bfly.i32
Fatal Python error: Aborted
```

判断当前是否满足：`python3 -c "import triton;print(triton.__version__)"` 应为 3.3.1（用户级）。
若显示 3.2.0，去掉 `--enable-triton-backend`（退回 eager，约 8.15 s/it，能跑但慢一倍）。

### 12-B. 量化选型：为什么是 int8，以及为什么拿不到加速

`comfy/quant_ops.py` 在 `torch.version.cuda < 13` 时 `registry.disable("cuda")`。本机 torch 是
cu128 → **cuda 后端被禁**（那里才有为 sm_120 编好的 int8/convrot/nvfp4 内核）。剩下 triton 与
eager。系统自带的 triton **3.2.0** 在 sm_120 上编译即崩，但**升到 triton 3.3.1 后 triton 后端
可用且快一倍**（§12-G 有完整实测）——故现行方案是 **int8 + triton 3.3.1**，不是 eager。

| 布局 | 能力枚举里有算子 | triton 3.2.0 时 | **triton 3.3.1 后（现行）** |
|---|---|---|---|
| int8 tensorwise | triton / cuda / eager | 仅 eager | **triton（4.09 s/it）** |
| w4a8 / fp8 | triton / cuda / eager | 仅 eager | triton（未实测） |
| int4 convrot w4a4 | cuda / eager（triton 无） | 仅 eager | 仍仅 eager（triton 无此算子） |
| **NVFP4** | cuda / eager | 仅 eager | 仍仅 eager（triton 无 nvfp4 linear） |

⚠️ **教训：能力枚举里有某个算子 ≠ 这块卡上编得出来。** 先前据 `list_backends()` 断言
"triton 有 int8_linear 所以 int8 有加速"，被 LLVM ERROR 推翻——枚举成立而编译失败。
反过来也要注意：**该结论只对当时的 triton 3.2.0 成立**，换版本后枚举同样的算子就真能用了。
**NVFP4 在本机始终无收益**：它的 linear 只在 cuda 后端，而 cuda 后端要 torch cu130，
本机驱动不允许（§12-G）。所以"5090 该上 NVFP4"这个直觉在这台机器上永远是错的。

加速的落地方案见 **§12-G**（triton 3.3.1，已做，2× 加速）；cu130 那条路为何不可行也在 §12-G。

### 12-C. 显存账（32G 的真正价值）

编码器 int8 **25.28 GiB** 整块驻显存成立：`loaded completely; 30396.67 MB usable, 25883.83 MB loaded, full load: True`。
这正是 24G 卡做不到、必须 offload 到 CPU 的地方（§9）。DiT 也是 `full load: True`（19996 MB）。
**DiT 必须用 pruned 档**：非 pruned int8 为 31.70 GiB > 31.4 GiB 可用显存，装不下。

### 12-D. 权重（全部取自官方 `Comfy-Org/MiniMax-H3`，ModelScope）

共 69.75 GiB，`safetensors.safe_open` 逐个验 header 通过，体积精确吻合：

| 文件 | GiB | 张量数 |
|---|---|---|
| `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | 25.28 | 1602 |
| `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 19.53 | 932 |
| `diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors` | 19.53 | 932 |
| `vae/minimax_h3_video_vae_fp16.safetensors` | 4.85 | 562 |
| `vae/minimax_h3_audio_vae_fp32.safetensors` | 0.56 | 917 |

**官方仓库无 §8 的 L2P 尾标记**（实测尾字节 = 0，无需剥离）；那个坑只在第三方 Abiray 仓库出现。
**两个来源命名不同、不可混用**：官方全小写 `minimax_h3_*` / 第三方大写 `MiniMax_H3_*`。
脚本已按来源区分（`DIT_VARIANTS`）。

### 12-E. 实测结果（四模态 + LoRA，全过）

eager 后端，20 步，5s → 124 帧（17k+5），864×480 @24fps + aac 立体声 32kHz：

| 发 | s/it | 总耗时 | ffprobe | 画面核验 |
|---|---|---|---|---|
| t2v / i2v / fl2va / r2v | 8.13–8.16 | 232–255s | 全过 | 与 prompt 相符 |
| lora_off / lora_on | 8.13–8.16 | ~245s | 全过 | 见下 |

对比 4090 int4 eager 约 5.75 s/it：**5090 int8 每步反而慢约 1.4 倍**（都是 eager，int8 要反量化的
位数是 int4 的两倍）。32G 买到的是"编码器不必 offload"，不是算力。

**LoRA 在 int8 convrot 量化 DiT 上确实生效**（legacy ModelPatcher 的静默失效担忧未成立）。证据链：
`/history` 显示 lora_on 图中有 `LoraLoaderModelOnly`（`lora_name=riding_pose_H3_i2v_v1.0.safetensors`,
`strength_model=1.0`, `model` 取自 UNETLoader `105:6`）而 lora_off 无该节点；两发 seed 均 777、
prompt 与首帧完全相同；产物 PSNR 平均 **11.99** / SSIM **0.721**（自身对自身基准为 `inf`）；
画面上 LoRA ON 输出的是马与骑手的**侧面骑乘姿态特写**，与该 LoRA 名称 `riding_pose` 吻合，构图完整无崩坏。

### 12-F. 本次实测暴露并修掉的两个真 bug（dry-run 抓不到）

**1. prompt 从未注入（最严重）。** prompt 有两个来源：核心节点（`MiniMaxH3ImageToVideo #104` /
`MiniMaxH3ReferenceToVideo #149`）自己的 `widget[0]`，以及该节点的 `prompt` 输入边（模板接到
QwenVL 增强器）。此前**没有任何一处写 `--prompt`**：`apply_params` 的 docstring 声明"prompt 属
task 7 不碰"，而 task 7 的 `rewire_core_prompt_to_raw` 只把输入边**重接**到子图边界、从不写值。
后果是 t2v/i2v/fl2va 实际渲染**模板预置的 vaporwave 片头示例**（`'Vaporwave title sequence look…'`，
画面里带 "COMFYUI" / "STARRING" / "LATENT" / "CONTROLNET" 字样），r2v 渲染**空串**。
已修：写 core `widget[0]` + 子图边界 + 增强器 三处，并在增强器旁路时**剪断 core.prompt 输入边**
以消除 link/widget 二源歧义；加硬断言（prompt 空则 rc=1 拒绝提交）；dry-run 现在落盘
`/tmp/run_h3_dryrun.json` 以便离线核对提交内容。

**2. R2V 未使用的参考 LoadImage 让整个 prompt 被拒。** 只给 1 张 `--ref-image` 时，`#152` 保留模板
里作者机器上的示例图名 `s02_ep06_unexpectedcall_371_v01.jpg`，服务端 combo 校验判
`unknown_enum_value` 并拒掉整个 prompt。已修：未提供对应输入的 LoadImage 一律旁路
（`#149` 的 `ref_images/ref_videos/ref_video_audios/ref_audios` 均为 optional，旁路安全）。
不能"塞一张已上传的图凑数"——那会凭空多一个参考图、改变结果。

⚠️ **教训：`ffprobe` 过了不等于内容对。** 上面两个 bug 产出的都是完全合规的 h264+aac、
帧数与分辨率全对的视频，只有**抽帧看画面**（以及查 `/history` 里服务端真正收到的 API prompt）
才能发现渲染的根本不是你要的东西。验收必须包含看图这一步。

另一个诊断技巧：ComfyUI 命中执行缓存（`Prompt executed in 6.34 seconds` + 产物字节数与上一发
完全相同）是"条件输入没变"的可靠信号——改了 prompt 却仍命中缓存，说明 prompt 没进到采样器。

### 12-G. 解开加速后端：triton 3.3.1（**已落地，2× 加速**）；cu130 此机不可行

**先说 cu130 为什么不能做**（别再尝试）：`cuda` 后端要 torch cu130，而 CUDA 13.x 要求 NVIDIA
驱动 **≥ 580.65**（官方文档）。本机驱动 **570.86.10**，`nvidia-smi` 报的 CUDA 上限就是
**12.8**，装了 cu130 的 torch 起不来。驱动是宿主机内核模块，容器内 sudo 管不到（租用机器）。
CUDA 前向兼容（`cuda-compat`）只对数据中心卡支持，本机是 GeForce，用不上。
额外风险：当前 `2.7.0.dev20250310+cu128` 是 nightly dev 构建，**pip 缓存里已无其 wheel**，
覆盖后很可能无法精确回滚——而它是唯一验证过能跑的组合。

**可行且已落地：用户级升 triton 到 3.3.1。** 关键在于 3.2.0 的失败根源是它**自带的 LLVM**
不认识 sm_120，而**不是**驱动或 CUDA 版本问题——CUDA 12.8 本身支持 sm_120，本机 ptxas 也支持。
换一个自带更新 LLVM 的 triton，不动驱动、不动 torch 即可。

```bash
pip install --user --no-deps triton==3.3.1 -i https://mirrors.aliyun.com/pypi/simple/
python3 -c "import triton,os;print(triton.__version__, os.path.dirname(triton.__file__))"
# 期望：3.3.1 /home/PER3cU/.local/lib/python3.10/site-packages/triton
```

`--no-deps` 是必须的：torch 声明的依赖是硬钉 `pytorch-triton==3.2.0+git4b3bb1f8`，不加会被拖动。
本工作负载不走 torch.compile / inductor（comfy_kitchen 直接 `@triton.jit` 调内核），故版本错配
影响面小——但这只能靠实测确认，见下面第 2 层证据。

**为什么这样装可以干净回滚**：`sys.path` 中 `~/.local/lib/python3.10/site-packages` 排在系统
`/usr/local/lib/python3.10/dist-packages` **之前**，用户级安装只是**遮蔽**系统的 3.2.0，系统包
原封不动。回滚 = 删掉用户级目录下的这两项，随后 `import triton` 会自动回到系统 3.2.0：

- `~/.local/lib/python3.10/site-packages/triton`
- `~/.local/lib/python3.10/site-packages/triton-3.3.1.dist-info`

**实测（三层证据，逐层收紧）**

1. **最小 triton 内核**（`tl.sum` 的 warp reduction，正是产生 `shfl.sync.bfly` 的地方，
   探针在 `~/triton_probe.py`）：3.2.0 精确复现 LLVM ERROR；**3.3.1 编译并执行成功、数值一致**。
2. **comfy_kitchen 自己的内核**（真正的 API 兼容风险点）：
   `backends.triton.quantization.int8_linear` 在 3.3.1 下可用，对 bf16 参考的相对误差
   **0.0126**（int8 量化噪声量级）；单算子 **triton 0.167 ms vs eager 0.353 ms = 2.11×**。
3. **真实渲染**（t2v，5s / 124 帧 / 20 步）：

| 后端 | s/it | 采样耗时 | 总耗时 |
|---|---|---|---|
| eager（triton 3.2.0） | 8.15 | 2:42 | 232–255 s |
| **triton 3.3.1** | **4.09** | **1:29** | **197 s** |

**画质无劣化**：与 eager 同 prompt 同 seed 42 的产物相比 PSNR **44.26 dB** / SSIM **0.9797**
（参考：无关内容约 15 dB，完全相同为 `inf`）——肉眼无法分辨，同一份 int8 数学的不同内核实现。
`LLVM ERROR` / `Fatal Python error` 计数为 0。

至此 5090 int8 的 **4.09 s/it 已快过 4090 int4 eager 的约 5.75 s/it**，32G 机的选型收益完整成立：
既拿到"25.28G 编码器不必 offload"，也拿到了算力。

### 12-H. ——upscale TRT 超分链路（**2026-08-08 验证通过**）

`--upscale` 走模板自带的 `Upscaler-Tensorrt` 超分链（node `#128` LoadUpscalerTensorrtModel → `#127` UpscalerTensorrt → VHS `#126` 出片 `H3up_<run_id>.mp4`）。实测：主片 480x640 / 243帧 / 10.12s，2x 超分产物 **960x1280** 正常生成（h264+aac，24fps），总约 5 分钟。工程师 node 首次 build ~43s 后缓存于 `models/tensorrt/upscaler/RealESRGAN_x4_fp16_..._<trt版本>.trt`（69M），后续直接复用。

**环境硬约束**：服务器装的是 **tensorrt-cu12 11.2.1.2**（TRT 11）。两个坎、两个节点补丁（都在服务器 `custom_nodes/ComfyUI-Upscaler-Tensorrt/`，备份 `.bak`）：

1. **TRT 11 删除了 `BuilderFlag.FP16`**（弱类型精度开关全移除）→ 老节点 `trt_utilities.py:217` 无条件求值 `trt.BuilderFlag.FP16` 直接 AttributeError（node 128）。补丁：改用 `hasattr(trt.BuilderFlag,"FP16")` 守卫，TRT 11 下跳过 → 引擎按 **fp32** 构建。代价：fp32 比 fp16 慢、且引擎上下文偏大（见下）。
2. **TRT 执行上下文需 ~17.6GB 显存**，而主扩散模型（int8 ~26GB）在**同一 prompt 内一直驻留** —— ComfyUI 的自动换载只管“加载别的模型时腾挪”，管不到 TensorRT 引擎的**原生 cudaMalloc**（模型管理器看不见它），32G 卡直接 OOM → `create_execution_context()` 返回 None → node 127 崩 `'NoneType' ... set_input_shape`。补丁：在 `nodes/upscaler_tensorrt.py` 的 `activate()/allocate_buffers()` **之前**插 `mm.unload_all_models()` + `mm.soft_empty_cache()`，先腾掉扩散模型再建上下文（超分已属后处理，之后无节点再需要模型，重载不会发生）。

**为什么没降级到 TRT 10 保 fp16（别再尝试）**：GFW 下拿不到 10.x 运行时库。PyPI 上 `tensorrt_cu12_libs` **永远只有 sdist 没有 wheel**，其 sdist 走 NVIDIA 自研 `wheel-stub` 后端（pyproject 里 `index_url=https://pypi.nvidia.com/`）去拉真正的大库；而 `pypi.nvidia.com` 从本机访问 `tensorrt_cu12_libs/`、`tensorrt_cu12_bindings/` **全 404**，连一体化 `tensorrt` 也只有 sdist。结论：fp32 已够用，降级被网络堵死。

**操作提醒**：给任意自定义节点 `.py` 打补丁后必须**重启 ComfyUI 才生效**（模块启动时已载入内存）。重启命令见 §12-A；重启顺带清掉上次残留的 ~26GB 驻留显存（`nvidia-smi` 验证空闲 <1GB 占用再跑）。

