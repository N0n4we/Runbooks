# MiniMax H3 全模态视频生成 — 部署与执行 Runbook

> 目标机：wp08.unicorn.org.cn:33307（XbbcY9）/ RTX 3090 24G / 157G RAM（cgroup 128G）/ ComfyUI 0.30.0（`~/comfy/ComfyUI`，8188，常驻 `--lowvram --reserve-vram 5`）
> 底模：Comfy-Org 官方打包 `Comfy-Org/MiniMax-H3`（pruned_int8_convrot 系）；底图：Comfy-Org 官方模板两枚（fl2va 文生模板 + ref2va 参考模板；i2v 由脚本动态接首/末帧图，无需独立模板）
> 一键脚本：`run-h3.py`（支持 H3 全部输入模态：文本 / 首末帧图 / ≤9 参考图 / ≤3 参考视频（带音轨）/ ≤3 独立音频；暂不含 LoRA）

## 0. 这是什么

MiniMax H3 = 全模态打包 DiT：文本/图像/视频/音频统一上下文理解 → 输出**带原生双声道音轨**的视频（人声/音效/配乐单次前向一起出，非事后叠加），最高 2K、24fps、约 15s。ComfyUI ≥0.30.0 原生支持（`comfy_extras/nodes_minimax_h3.py` 四节点），**零第三方节点包**即可跑。

- 两个底模家族：**fl2va**（T2V 纯文本 + I2V 首/末帧）与 **ref2va**（参考驱动：图/视频/音频混合条件），权重不通用。
- 核心节点 `MiniMaxH3ImageToVideo`（t2va/fl2va：prompt + 可选 first/last_frame）与 `MiniMaxH3ReferenceToVideo`（ref2va：动态槽 ref_image_0..8 / ref_video_0..2 / ref_video_audio_0..2 / ref_audio_0..2）。
- 无负面词、无 CFG（BasicGuider=1）；采样 res_multistep + simple 20 步为官方模板默认。
- 帧数网格 **17k+5 @24fps**（5s=124 帧），训练范围 124~362 帧（≈5~15s）；画布 32 倍数，原生 768 短边、面积上限 768×1344。

## 1. 环境与网络（wp08:33307 实测，2026-08-04）

- **本机 codeload.github.com 直连可达（~5MB/s）**——与旧机的「github 直连死」不同，ComfyUI 主程序直接官方 tar：
  ```bash
  mkdir -p ~/comfy && cd ~/comfy
  curl -sL -o ComfyUI.tar.gz "https://codeload.github.com/comfyanonymous/ComfyUI/tar.gz/refs/heads/master"
  gzip -t ComfyUI.tar.gz && tar xzf ComfyUI.tar.gz && mv ComfyUI-master ComfyUI
  pip install -r ComfyUI/requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --retries 30 --timeout 60
  pip install comfy-cli modelscope -i https://mirrors.aliyun.com/pypi/simple/
  ```
- 裸容器无 curl/ffmpeg/git：`sudo apt-get install -y curl wget ffmpeg libgl1 unzip git`（apt 源已是阿里；后台 apt 被 kill 后 dpkg 锁会自动释放，直接重跑即可）。**git 必装**：`comfy run` 的 GitPython 无 git 直接拒绝执行。
- 模型走 ModelScope CLI（~12MB/s、自动续传）：`modelscope download --model Comfy-Org/MiniMax-H3 "<repo路径>" --local_dir ~/comfy/ComfyUI/models`（目录结构自动对齐）。
- torch 2.7.0.dev20250310+cu128（镜像预装，<2.8 → legacy ModelPatcher，官方核心节点正常）。
- 启动（**`--reserve-vram 5` 必需**，理由见 §5）：
  ```bash
  setsid nohup python3 main.py --listen 0.0.0.0 --port 8188 --disable-auto-launch \
    --lowvram --reserve-vram 5 > /tmp/comfy_boot.log 2>&1 </dev/null & disown
  ```
- cgroup 内存上限 128G（`memory.max`），与旧机同；qwen int8 27G 走 CPU fp16 编码进 RAM，单发峰值远低于上限。
- 验证：`/object_info` 应有 MiniMaxH3ImageToVideo / MiniMaxH3ReferenceToVideo / MiniMaxH3SigmaShift / EmptyMiniMaxH3LatentAV（全量 827 节点）。

## 2. 模型文件（24G 选型）

| 目标路径 | 大小 | 说明 |
|---|---:|---|
| `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 20.97G | T2V/I2V 底模（官方模板默认精度） |
| `diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors` | 20.97G | R2V 底模 |
| `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | 27.14G | **必须选 int8**：模板默认的 nvfp4_awq 是 Blackwell(sm_120) 专用，3090(sm_86) 不能用；bf16 51.5G 无必要 |
| `vae/minimax_h3_video_vae_fp16.safetensors` | 5.21G | 视频 VAE |
| `vae/minimax_h3_audio_vae_fp32.safetensors` | 0.61G | 音频 VAE（原生音轨） |

合计 ~75G。仓库另有 bf16(66G)/int8_convrot 全量(34G)/pruned_fp8_scaled(21G) 备选；pruned int8 是官方模板默认值且 Bernini 已验证 int8_convrot 路径在本机 torch 上可用。CLIPLoader 的 type 必须保持 `minimax`。

## 3. 工作流结构（两模板，仓库内 `H3文生视频工作流.json` / `H3参考生视频工作流.json`）

> 官方模板库本有三枚（t2v/i2v/r2v），但 t2v 与 i2v 模板的子图定义逐字段一致（只差 i2v 多一个 LoadImage 和一条未接线装饰链），脚本用文生模板 + 动态接首/末帧即可覆盖 i2v，故只留两枚。

**fl2va 文生模板**（子图 `4c314f31` 实例 105，promoted widgets 权威）：

```
顶层: ResolutionSelector(115) → 105.width/height；脚本按需新建 LoadImage → 105.first_frame/last_frame
子图内: UNETLoader 6(fl2va) / CLIPLoader 13(qwen3vl, type=minimax) /
        VAELoader 11(视频) 24(音频) / RandomNoise 15 / KSamplerSelect 17(res_multistep) /
        BasicScheduler 9(simple,20,1) / PrimitiveFloat 111(秒) → ComfyMathExpression 107
        (17k+5 吸附) → MiniMaxH3ImageToVideo 104 → BasicGuider 16 → SamplerCustomAdvanced 14
        → VAEDecode 10 + VAEDecodeAudio 23 → CreateVideo 91(24fps)
105 → SaveVideo(92) → output/*.mp4
```

- 105.widgets_values = [prompt, width, height, 秒, seed, unet, clip, 视频vae, 音频vae]（instance 代理值优先于内部 widgets，**补丁要双写**）。
- links 格式陷阱：顶层 `links` 是**列表**，子图 `definitions.subgraphs[].links` 是**dict**——脚本手术两套都要兼容。

**R2V 平铺模板**：

```
UNETLoader 127(ref2va) / CLIPLoader 128 / VAELoader 119,120 → MiniMaxH3ReferenceToVideo 136
  ↑ LoadImage 137/139 → ref_images.ref_image_0/1（动态槽，shape=7，≤9）
  ↑ LoadVideo → GetVideoComponents.IMAGE → ref_videos.ref_video_k（≤3）
  ↑ GetVideoComponents.AUDIO → ref_video_audios.ref_video_audio_k（视频自带音轨）
  ↑ LoadAudio → ref_audios.ref_audio_j（≤3）
PrimitiveStringMultiline 138(prompt) / ResolutionSelector 115 / PrimitiveFloat 132(秒)
  → ComfyMathExpression 131 → 136.length
136 → BasicGuider 126 → SamplerCustomAdvanced 125（KSamplerSelect 123 + BasicScheduler 124 + RandomNoise 129）
→ VAEDecode 122 + VAEDecodeAudio 121 → CreateVideo 130(24fps) → SaveVideo 92
```

- 参考标签（提示词里引用，1 起始）：图片 `<Picture i>`、视频 `<Video k>`、音频 `<Audio j>`；**视频自带音轨先占位**（按视频顺序 <Audio 1..m>），独立音频续号。节点源码注释即此约定。
- 136 widgets = [prompt(被138接管), width, height, length, ref_image_size]；ref_image_size: `match`=参考图缩到画布面积（快）/ `max`=2048 短边（身份保真，慢数倍）。
- 参考视频要求 ≥5 帧且会吸附 17k+5、截到不长于成片帧数；Qwen 以 2fps 带时间戳看参考视频。

**SigmaShift（可选，两模板通用）**：`MiniMaxH3SigmaShift`（shift_video 默认 12 / shift_audio 默认 3）不在官方模板里，节点内建值生效。脚本的 `--shift-video/--shift-audio` 会在 UNETLoader 后动态插入该节点并**接管 UNET 的全部 MODEL 出边**（BasicGuider 与 BasicScheduler 必须共用同一份 patched model，否则调度表与 DiT 内部 video/audio shift 不一致）。

## 4. 一键脚本 run-h3.py

```bash
# T2V
python3 run-h3.py --prompt "……"
# I2V（首帧/末帧，可单用）
python3 run-h3.py --prompt "……" --image first.png [--last-image last.png]
# R2V（任意混合，≤9图/≤3视频/≤3音频）
python3 run-h3.py --prompt "让 <Picture 1> 在 <Video 1> 场景里，用 <Audio 2> 的嗓音……" \
    --ref-image role.png --ref-video scene.mp4 --ref-audio voice.mp3
# 常用：--seconds 5（17k+5 吸附）/ --frames 124 / --size 768x1344（32 倍数）
#       --aspect "16:9 (Widescreen)" --megapixels 0.4（原生画质 1.0≈1344x768）
#       --steps 20 / --sampler res_multistep / --scheduler simple / --seed N
#       --shift-video 12 --shift-audio 3（SigmaShift，缺省不插节点）
#       --ref-image-size match|max / --out x.mp4 / --no-download
```

行为：模态互斥校验 → 上传输入（run_id 前缀）→ 内存补丁共享 json（CLIP 换 int8、模型名、种子、采样、时长、画布、动态槽接线）→ `comfy run` 提交 → run_id 隔离轮询 → 拉回 ffprobe 验证（视频帧数 + **音轨存在性**）。输出 h264+aac mp4（QuickTime 可播）。

## 5. 验证记录（wp08:33307 / 3090 24G / --lowvram --reserve-vram 5，2026-08-04）

**`--reserve-vram 5` 的由来（关键教训）**：仅 `--lowvram` 时，legacy ModelPatcher（torch 2.7 <2.8 无 DynamicVRAM）把 21G int8 DiT **整载**进 24G VRAM（`loaded completely 19996 MB`），剩 ~4G 不够 int8_linear 反量化 `torch.cat` 暂存 + 60k token 注意力激活 → 采样器 OOM。加 `--reserve-vram 5` 后 DiT 变部分加载（14.8G 驻留 / 5.2G offload 流式），四发全过。

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
| SamplerCustomAdvanced OOM，栈落 `int8_linear→torch.cat`，boot log 见 DiT `loaded completely ~20G` | legacy ModelPatcher 把 21G DiT 整载进 24G，无暂存余量 → **启动加 `--reserve-vram 5`** 强制部分加载流式（§5） |
| `comfy run` 提交即败，报 git/GitPython | 裸容器无 git → `apt-get install git` |
| 官方 i2v 模板校验 `required input 'image' is missing`（节点 119） | i2v 模板 119/120 是未接线装饰链 → **别用 i2v 模板**，脚本文生模板 + 动态接帧已覆盖；手动跑官方 i2v 模板需先删 119/120 |
| CLIPLoader 报 nvfp4 相关错 | 模板默认 qwen3vl nvfp4_awq 是 Blackwell 专用 → 换 int8_convrot（脚本已自动） |
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
