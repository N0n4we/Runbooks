# SCAIL-2 角色动画工作流 — 部署与执行 Runbook

> 目标：在干净 Linux GPU 机器上，把 SCAIL-2 工作流部署跑通，生成视频。
> 作者原版**只启用 1 个 LoRA（Remix）**，其余 LoRA（lightx2v/DPO/slop）与 Uni3C 均为 bypass，不加载。Remix 为作者私有网盘文件，需单独获取后放入 `loras/`。
> 缺 Remix 时脸型/画风不对，是已知待补项（非动作僵硬问题）。

---

## 0. 环境与目标（重要：本 runbook 的主流程）

**目标**：在空 Linux GPU 服务器上，把 SCAIL-2 工作流部署跑通，生成视频。

**关键约束（已实测验证，wp08:19087 / RTX 4090 24G）**：

- **ComfyUI 主程序用 `comfy install` 会卡死**——它内部走 github clone，而本机 **github 直连不可达**（0.02MB/s 且 clone 半截失败）。实测可用的是 **`ghfast.top` 镜像拉 tar**（见第 3 节），或 `comfy-cli` 包本身用 `pip install comfy-cli` 装。
- **`comfy` 命令来自独立 PyPI 包 `comfy-cli`**（实测 1.12.0），提供 `comfy install` / `comfy run` 等。ComfyUI 主程序本身不提供 `comfy` 命令。
- **原版前端 json 不能"不改 json 直接 `comfy run` 直跑"**——实测 `comfy run --workflow 原版.json` 报 **14 个 validation error**，全是前端虚拟/UI 节点（`Int`、`Image Blank`、`Fast Groups Bypasser (rgthree)`、`Label (rgthree)`）服务端不认（见第 7 节详述）。这些节点在作者自己的 ComfyUI 网页前端里会被正确转换，但 `comfy run` 的客户端转换做得不全。
  → **要跑通必须改 json**：至少把 455/471 的文件名改成你有的素材名，并把那批虚拟节点类型映射成真实节点（见第 4 节）。"零改动直跑"在本机不成立。
- **Remix LoRA 由用户自备**（作者私有网盘），其余模型从 hf-mirror 下。

**环境前提**：
- Linux + NVIDIA GPU（实测 RTX 4090 24G；3090 24G 也行但需降分辨率/帧数）
- 有 `sudo`、`pip`、`python3.10+`
- 能访问 `hf-mirror.com`（HuggingFace 直连被墙，走镜像）
- `ghfast.top` / `gh-proxy.com` 镜像可用（github 直连死，但这两个镜像 0.8MB/s 通）
- 约 40GB 磁盘放模型

验证：
```bash
nvidia-smi
python3 --version        # 需 3.10+
```

---

## 1. 安装 comfy-cli（独立包，提供 comfy 命令）

```bash
pip install comfy-cli
comfy --version          # 应输出版本号，如 1.12.0
```

> 不要指望 `comfy install` 装 ComfyUI 主程序——本机它内部 github clone 会卡死。ComfyUI 主程序改用第 3 节的 ghfast 拉 tar 法。

---

## 2. 安装 ComfyUI 主程序（ghfast 镜像拉 tar，绕过 github 死链）

> `comfy install` 走 github clone，本机不可达 → 卡死。改用 `ghfast.top` 拉官方 release tar（与 image-generation runbook 同思路）。

```bash
cd ~
rm -rf ~/comfy ComfyUI 2>/dev/null
mkdir -p comfy && cd comfy
# ghfast 镜像拉 master tar（gzip -t 校验完整性，防截断）
wget -q --tries=3 "https://ghfast.top/https://github.com/comfyanonymous/ComfyUI/archive/refs/heads/master.tar.gz" -O ComfyUI.tar.gz
gzip -t ComfyUI.tar.gz && echo "GZIP OK" || echo "GZIP FAIL"
tar xzf ComfyUI.tar.gz && mv ComfyUI-master ComfyUI && echo "unpack OK"
# 装主程序依赖（阿里源，retry-loop 防超时）
pip install -r ComfyUI/requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --retries 30 --timeout 60
```

装完确认（关键：主 tar 含 `comfy/ldm/models/`，不缺 submodule）：
```bash
ls ~/comfy/ComfyUI/comfy/ldm/models/autoencoder.py   # 应存在
```

装系统依赖（LayerStyle / VideoHelperSuite 导入需要 libGL）：
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libgl1
```

装 SageAttention（工作流 attention_mode=sageattn 必需）：
```bash
pip install sageattention
python3 -c "from sageattention import sageattn; print('sageattn OK')"
```

> **Blackwell（sm_120，RTX 5090）额外步骤**（wp08:15024 实测）：
> - PyPI 上 sageattention 最高只有 **1.0.6**（triton 实现）；镜像/官方源都没有 2.x。它靠 triton 跑，而机器自带的 **triton 3.2 不认 sm_120**（`'sm_120' is not a recognized processor` / `LLVM ERROR: Cannot select`）→ **`pip install -U triton`**（≥3.3，实测 3.7.1 OK），装完再验 `sageattn(q,k,v)`。
> - 裸机**没有 nvcc**。pip 的 `nvidia-cuda-nvcc-cu12` 轮子**缺 nvcc 本体**（只有 ptxas，阿里源/pypi 一样）。解法：从 NVIDIA redist 拉编译器（`developer.download.nvidia.com` 实测可达）：
>   ```bash
>   cd /tmp && wget "https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.8.93-archive.tar.xz" -O nvcc.tar.xz
>   tar xf nvcc.tar.xz
>   # 拼 CUDA_HOME：bin/nvvm 指 redist，include/lib64 合并 pip 的 cuda_runtime/cccl/cuda_nvcc 头文件
>   ```
>   装 pip 件：`nvidia-cuda-nvcc-cu12==12.8.93 nvidia-cuda-runtime-cu12==12.8.90 nvidia-cuda-cccl-cu12==12.8.90`，`CUDA_HOME` 下 `include/`、`lib64/` 用 `cp -rs` 合并各包内容。编译类安装用 `CUDA_HOME=~/cuda128 TORCH_CUDA_ARCH_LIST="12.0" pip install ... --no-build-isolation`。
> - 5090 镜像已预装 torch 2.7.0+cu128（sm_120 matmul 正常），不用动 torch。

---

## 3. 安装 custom nodes（ghfast 拉 tar，不 git clone）

> 节点 github 直连死，用 `ghfast.top` 拉每个节点的 `archive/refs/heads/main.tar.gz`，gzip 校验后解包。
> 实测 8/10 成功：WanAnimatePlus / WanVideoWrapper / KJNodes / VideoHelperSuite / FeiHou-Toolbox / rgthree / essentials / Custom-Scripts 均 OK。
> LayerStyle + LayerStyle_Advance（Acly 仓库）在 ghfast 上 404，暂跳过——主链路（SCAIL2ColoredMask 在 FeiHou-Toolbox 里）不依赖它。

```bash
cd ~/comfy/ComfyUI/custom_nodes
G="https://ghfast.top/https://github.com"
declare -A nodes=(
  [ComfyUI-WanAnimatePlus]="wuwukaka/ComfyUI-WanAnimatePlus"
  [ComfyUI-WanVideoWrapper]="kijai/ComfyUI-WanVideoWrapper"
  [ComfyUI-KJNodes]="kijai/ComfyUI-KJNodes"
  [ComfyUI-VideoHelperSuite]="Kosinkadink/ComfyUI-VideoHelperSuite"
  [ComfyUI-FeiHou-Toolbox]="FX-FeiHou/ComfyUI-FeiHou-Toolbox"
  [rgthree-comfy]="rgthree/rgthree-comfy"
  [ComfyUI_essentials]="cubiq/ComfyUI_essentials"
  [ComfyUI-Custom-Scripts]="pythongosssss/ComfyUI-Custom-Scripts"
)
for d in "${!nodes[@]}"; do
  repo=${nodes[$d]}
  url="$G/$repo/archive/refs/heads/main.tar.gz"
  wget -q --tries=2 "$url" -O "$d.tar.gz" && gzip -t "$d.tar.gz" 2>/dev/null && {
    tar xzf "$d.tar.gz" && mv "$d-main" "$d" && echo "OK $d"
  } || echo "FAIL $d"
done
# 装各节点依赖（每个节点目录的 requirements.txt，阿里源）
for d in */; do
  [ -f "$d/requirements.txt" ] && pip install -r "$d/requirements.txt" -i https://mirrors.aliyun.com/pypi/simple/ --retries 20 --timeout 60
done
```

> 节点依赖缺 `accelerate`/`cv2`(opencv) 会导致节点 MISSING（import 失败）。装完 requirements 后重启服务，用 `/object_info` 查 `WanAnimatePlus` / `SCAIL2ColoredMask` / `VHS_LoadVideo` / `KJNodes` 是否注册（共约 1395 节点）。

---

## 4. 工作流文件（必须改，不能原样直跑）

原版前端 json（如 `Scail-2+高阶工作流Plus+V4版【长视频+++多图参考+++动作迁移】.json`）上传到服务器（如 `~/scail_workflow.json`）。

**必须做服务端兼容修改才能跑**（实测 `comfy run` 原样直跑报 14 个 validation error；后续执行期还会撞 schema/显存问题）：

1. **455 LoadImage 单图文件名**：作者原写 `13e6b9...png`（哈希私有名）→ 改成你有的 `06_left.png`
2. **471 VHS_LoadVideo 驱动视频名**：作者原写 `d21edfd...mp4`（哈希私有名）→ 改成你有的 `ref.mp4`
3. **虚拟/UI 节点类型映射**（实测 `comfy run` 的客户端转换不做这个，服务端不认 → 13 个 `unknown_class_type` 错误）。

   原版 json 里这些节点**有/没有默认值**（已从 json 扒出实测）：

   | 前端 type | 节点 id | widgets_values(实测) | 改法 | 说明 |
   |------|---------|---------------------|------|------|
   | `Int` | 457 / 479 | `["0"]` | type→`PrimitiveInt`，**值照抄 0** | 整数控件，mode=0 启用，没被 link 连，独立填值 |
   | `Image Blank` | 506 | `[512, 8, 0, 0, 0]` | type→`EmptyImage`，**新版必须改 `[512,8,1,0]`**（width,height,batch_size=1,color=0 黑图占位） | ⚠️ 新版 ComfyUI（0.28 实测）EmptyImage 第 3 个 widget 是 **batch_size（min=1）**，照抄 `[512,8,0,0,0]` 会得到 batch_size=0 → 校验失败**静默跳过输出节点**，几十秒"假成功"不出片。width/height 实际被 link 583/584 从 487/493 连入，自带值是 fallback |
   | `Fast Groups Bypasser (rgthree)` | 529/530/533/581 | `null` | type→`FastGroupsBypassSwitch`，**保留 mode 与连线不动** | 分组开关，作者用来开关多图参考/Uni3C 等；null 是前端开关态，映射后由真实节点接管 |
   | `Label (rgthree)` | 535/536/537/538/539/540 | `null` | **直接删掉该节点 + 删连它的 link** | 纯画布文字标签，不参与计算，服务端无对应节点，主链路不需要 |

   > `Int`/`Image Blank` **有默认值**，改 type 时数值原样抄；`FastGroupsBypassSwitch` 映射后保留它的连线（它输出连到下游 mode=4 节点）；`Label` 6 个纯装饰，删节点 + 删 link 即可。
   > 这些虚拟节点在作者网页前端里会被自动翻译成真实节点；`comfy run` 漏了这步，所以手动改 json。
4. **576 SCAIL_2 Embeds 的 BOOL schema**：`widgets_values` 里 `single_frame_prefix_encoding` 若仍是旧 enum 字符串 `"disabled"`，新版节点定义是 BOOLEAN，要改成 JSON `false`（不是字符串 `"false"`）。这就是 `shape_mismatch ... expected BOOL received str` 的根因。
5. **无 Remix 的降级跑法**：580 LoraSelect 不能填 `none`（下拉无此枚举，会 `unknown_enum_value`）。缺 Remix 时把 580 节点 `mode` 改成 `4`（bypass），其余结构不动；等拿到 Remix 后恢复 `mode=0` 并填文件名。
6. **RTX 4090 24G 实测稳定参数**：驱动视频 `ref.mp4` 若是 922 帧，479 `frame_load_cap` 从 `0` 改成 `[161]`；490 `SCAIL2ColoredMaskV2` 的 `render_device` 改成 `'cpu'`；252 PlaySound 可 `mode=4` bypass。否则会在 490 CUDA OOM，或整个 ComfyUI 进程被 CPU 大 tensor 拖死。
7. **生成尺寸自动跟随参考图**（2026-07-23 起入共享配置）：新增 `GetImageSize` 节点（id=900），`455 LoadImage.IMAGE` → `900.image`，`900.width` → 525 SetNode、`900.height` → 524 SetNode，拆掉 531/532 INTConstant 的连线（节点保留，可手动回退）。此后画布自动 = 参考图尺寸（576 内部对 32 向下取整，如 896×1200 → 896×1184），换参考图不用再手改 531/532。动机与证据见第 9 节“成片角色被拉伸/压扁”。

> 这些改动属于**服务端兼容/24G 稳定参数**：文件名、虚拟节点 type 映射/删除、schema 类型、缺 Remix 时 bypass、帧数上限与 490 CPU 渲染；不改主链路结构/连线。
> 之前手写 `_frontend_to_api.py` 转换脚本也是在干这事，但容易弄断 Get/Set/尺寸连线导致比例错；改 json 本身更可控。

---

## 5. 准备输入素材

把参考图 / 驱动视频放进 ComfyUI 的 input 目录（注意 `comfy install`/`ghfast` 的路径是 `~/comfy/ComfyUI`）：

```bash
mkdir -p ~/comfy/ComfyUI/input
# 文件名需匹配工作流里写的（LoadImage→06_left.png，VHS_LoadVideo→ref.mp4）
# 不改 json 则改本地文件名去匹配：把你有的图改名 06_left.png、视频改名 ref.mp4 传上去
cp /你的/参考图.png ~/comfy/ComfyUI/input/06_left.png
cp /你的/驱动视频.mp4 ~/comfy/ComfyUI/input/ref.mp4
```

---

## 6. 启动 ComfyUI 服务

```bash
cd ~/comfy/ComfyUI
nohup python3 main.py --listen 0.0.0.0 --port 8188 --disable-auto-launch > /tmp/comfy_boot.log 2>&1 &
# 等 API 起来
for i in $(seq 1 40); do
  python3 -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8188/object_info',timeout=5)" 2>/dev/null && break
  sleep 3
done
python3 -c "import urllib.request;print('API',urllib.request.urlopen('http://127.0.0.1:8188/object_info',timeout=10).status)"
```

---

## 7. 提交执行（comfy run，需先按第 4 节改 json）

```bash
# 服务已在 8188 跑着；喂改好的前端 json
comfy run --workflow ~/scail_workflow.json --host 127.0.0.1 --port 8188 --wait
```

- `--workflow` 接受前端 UI 格式或 API 格式，前端格式客户端自动转 API（但虚拟节点转换不全，见第 4 节）
- `--wait` 阻塞到执行完（不带则提交即返回）
- `--verbose` 看采样进度
- 输出在 `~/comfy/ComfyUI/output/<日期>/<时间>_Wanimate_00001.mp4`

**实测 `comfy run` 原样直跑原版 json 的报错（14 个 error）**：
- 13 个 `unknown_class_type`：`Int` / `Image Blank` / `Fast Groups Bypasser (rgthree)` / `Label (rgthree)` 服务端不认 → 第 4 节虚拟节点映射解决
- 1 个 `shape_mismatch`：576 节点 `single_frame_prefix_encoding` 期望 BOOL 收到 str → 改 json 该 widget 值类型
- 2 个 `unknown_enum_value`（455/471 文件名）：作者哈希私有名不在 input/ → 第 4 节改文件名解决

不改这些，`comfy run` 直接失败。改完后再跑即通过校验、进入采样。

不阻塞提交（提交后自己轮询；wp08 没有 `curl`，用 Python urllib）：
```bash
comfy run --workflow ~/scail_workflow.json --host 127.0.0.1 --port 8188
python3 - <<'PY'
import urllib.request
for path in ('queue','history'):
    with urllib.request.urlopen(f'http://127.0.0.1:8188/{path}',timeout=10) as r:
        print(path, r.status, r.read(500).decode('utf-8','ignore'))
PY
```

- wp08 / RTX 4090 24G 实测：**161 帧、6 步、3 个 chunk ≈ 19 分 49 秒**；每步约 55–73 秒。更长视频会近似随 chunk 数线性增加。
- 显存峰值约 17.6GB（161 帧、704×1280 输出、block swap 30）；922 帧整段会在 490 遮罩渲染阶段 OOM/杀进程，必须先按第 4 节限制 `frame_load_cap`。
- 输出文件名含 `%date:...%` 占位符未展开属正常；本轮成功产物实际在 `output/%date:yyyy-MM-dd%/%date:yyyyMMdd_hhmmss%_Wanimate_00001-audio.mp4`。

下载到本地（注意本轮输出目录名是 literal `%date:yyyy-MM-dd%`）：
```bash
scp -P <端口> -o StrictHostKeyChecking=accept-new \
  'user@host:/home/user/comfy/ComfyUI/output/%date:yyyy-MM-dd%/*Wanimate_00001*.mp4' \
  ~/Downloads/
```

---

## 8. 下载模型（全部从 hf-mirror；URL 以 tree API 实测为准）

> **关键更正**：`Comfy-Org/SCAIL-2` 仓库只有 `diffusion_models/` 和 `loras/`，**没有 VAE/T5/CLIP**。VAE 来自 `Comfy-Org/Wan_2.1_ComfyUI_repackaged` 的 `split_files/`；T5 必须用**非 scaled** fp8；CLIP-ViT-H 用完整 2.53G 文件；SAM3 在 `checkpoints/` 子路径，直接放进 `models/checkpoints/`（不要再用 detection+软链）。
>
> 列仓库实际文件的可靠方法（hf-mirror 的仓库主页/`/api/models/{repo}` 常 403）：
> ```bash
> python3 - <<'PY'
> import urllib.request,json
> repo='Comfy-Org/SCAIL-2'
> url=f'https://hf-mirror.com/api/models/{repo}/tree/main?recursive=true'
> req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0','Accept':'application/json'})
> data=json.load(urllib.request.urlopen(req,timeout=60))
> for x in data: print(x.get('path'), x.get('size'))
> PY
> ```

### 正确模型 URL → 目标路径（目标文件名必须与工作流一致）

| 目标路径 | 实际下载 URL | 大小 |
|---|---|---:|
| `models/diffusion_models/wan2.1_14B_SCAIL_2_fp8_scaled.safetensors` | `https://hf-mirror.com/Comfy-Org/SCAIL-2/resolve/main/diffusion_models/wan2.1_14B_SCAIL_2_fp8_scaled.safetensors` | 17,694,586,857 |
| `models/vae/Wan2_1_VAE_bf16.safetensors` | `https://hf-mirror.com/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors` | 253,815,318 |
| `models/text_encoders/umt5-xxl-enc-fp8_e4m3fn.safetensors` | `https://hf-mirror.com/realung/umt5-xxl-enc-fp8_e4m3fn.safetensors/resolve/main/umt5-xxl-enc-fp8_e4m3fn.safetensors` | 6,731,333,792 |
| `models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors` | `https://hf-mirror.com/sharing1179/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors/resolve/main/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors` | 2,528,373,448 |
| `models/checkpoints/sam3.1_multiplex_fp16.safetensors` | `https://hf-mirror.com/Comfy-Org/SAM3/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors` | 1,745,546,848 |

> **T5 注意**：不要用 Wan repack 的 `umt5_xxl_fp8_e4m3fn_scaled.safetensors` 顶替；节点 289 `WanVideoTextEncodeCached` 会报 `double-valued cfg ... non-scaled fp8 model`。工作流要的是**非 scaled** 版本。
> **CLIP 注意**：Wan repack 的 `split_files/clip_vision/clip_vision_h.safetensors` 只有 1.26G（视觉权重子集）；本轮跑通用的是 `sharing1179` 仓的完整 `CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors` 2.53G。

### 可续传 Range 分段下载（推荐）

hf-mirror 单流约 0.5–0.8MB/s；服务器支持 HTTP Range（206），用 6–8 个分段并行能显著提速。**不要多个进程同时写同一个最终 safetensors**；每段写自己的 `.part`，全部完成后校验大小并原子合并。

```bash
cat > /tmp/scail_range_download.py <<'PY'
import os,time,urllib.request,concurrent.futures
BASE=os.path.expanduser('~/comfy/ComfyUI/models')
ITEMS=[
 ('https://hf-mirror.com/Comfy-Org/SCAIL-2/resolve/main/diffusion_models/wan2.1_14B_SCAIL_2_fp8_scaled.safetensors','diffusion_models/wan2.1_14B_SCAIL_2_fp8_scaled.safetensors',17694586857,8),
 ('https://hf-mirror.com/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors','vae/Wan2_1_VAE_bf16.safetensors',253815318,4),
 ('https://hf-mirror.com/realung/umt5-xxl-enc-fp8_e4m3fn.safetensors/resolve/main/umt5-xxl-enc-fp8_e4m3fn.safetensors','text_encoders/umt5-xxl-enc-fp8_e4m3fn.safetensors',6731333792,8),
 ('https://hf-mirror.com/sharing1179/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors/resolve/main/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors','clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors',2528373448,6),
 ('https://hf-mirror.com/Comfy-Org/SAM3/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors','checkpoints/sam3.1_multiplex_fp16.safetensors',1745546848,6),
]

def download_file(url,out,total,workers):
    print('FILE_START',out,flush=True)
    partdir=out+'.ranges'; os.makedirs(partdir,exist_ok=True); os.makedirs(os.path.dirname(out),exist_ok=True)
    chunk=(total+workers-1)//workers
    def one(i):
        start=i*chunk; end=min(total-1,(i+1)*chunk-1); expected=end-start+1
        final=os.path.join(partdir,f'{i:02d}.part'); tmp=final+'.tmp'
        if os.path.exists(final):
            if os.path.getsize(final)==expected:
                print('PART_SKIP',os.path.basename(out),i,flush=True); return
            raise RuntimeError(f'final part size mismatch {final}')
        for attempt in range(1,20):
            try:
                have=os.path.getsize(tmp) if os.path.exists(tmp) else 0
                if have>expected: os.remove(tmp); have=0
                if have==expected:
                    os.replace(tmp,final); print('PART_DONE',os.path.basename(out),i,flush=True); return
                req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0','Range':f'bytes={start+have}-{end}'})
                with urllib.request.urlopen(req,timeout=180) as r,open(tmp,'ab') as f:
                    cr=r.headers.get('Content-Range','')
                    if r.status!=206 or not cr.startswith(f'bytes {start+have}-{end}/'):
                        raise RuntimeError(f'bad response status={r.status} range={cr}')
                    n=have; last=time.time()
                    while True:
                        b=r.read(8*1024*1024)
                        if not b: break
                        f.write(b); n+=len(b)
                        if time.time()-last>=20:
                            print(f'{os.path.basename(out)} part {i} {n/1e9:.2f}/{expected/1e9:.2f}GB',flush=True); last=time.time()
                if os.path.getsize(tmp)!=expected: raise RuntimeError(f'size {os.path.getsize(tmp)} != {expected}')
                os.replace(tmp,final); print('PART_DONE',os.path.basename(out),i,flush=True); return
            except Exception as e:
                print(f'RETRY {os.path.basename(out)} part {i} attempt {attempt}: {e!r}',flush=True)
                time.sleep(min(60,attempt*3))
        raise RuntimeError(f'{out} part {i} failed')
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one,range(workers)))
    join=out+'.join.tmp'
    with open(join,'wb') as dst:
        for i in range(workers):
            p=os.path.join(partdir,f'{i:02d}.part'); expected=min(total-1,(i+1)*chunk-1)-i*chunk+1
            if not os.path.exists(p) or os.path.getsize(p)!=expected: raise RuntimeError(f'bad part {p}')
            with open(p,'rb') as src:
                while True:
                    b=src.read(64*1024*1024)
                    if not b: break
                    dst.write(b)
    if os.path.getsize(join)!=total: raise RuntimeError('joined size mismatch')
    os.replace(join,out)
    print('FILE_DONE',out,os.path.getsize(out),flush=True)

for url,rel,total,workers in ITEMS:
    download_file(url,os.path.join(BASE,rel),total,workers)
print('ALL_MODELS_DONE',flush=True)
PY
setsid python3 /tmp/scail_range_download.py >/tmp/scail_range_download.log 2>&1 </dev/null &
disown
# 查进度
tail -f /tmp/scail_range_download.log
```

下载后逐个校验：

```bash
python3 - <<'PY'
from safetensors import safe_open
from pathlib import Path
files=[
 'diffusion_models/wan2.1_14B_SCAIL_2_fp8_scaled.safetensors',
 'vae/Wan2_1_VAE_bf16.safetensors',
 'text_encoders/umt5-xxl-enc-fp8_e4m3fn.safetensors',
 'clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors',
 'checkpoints/sam3.1_multiplex_fp16.safetensors',
]
base=Path.home()/'comfy/ComfyUI/models'
for rel in files:
    p=base/rel
    with safe_open(str(p),framework='pt') as f:
        print('OK',rel,p.stat().st_size,len(f.keys()))
PY
```

### 待补 LoRA（唯一一个，作者 FX-FeiHou 私有网盘，非公开）

- 作者原版唯一启用的 LoRA（580 节点，mode=0，**用户自备，完整文件名不录入本文档**）—— 决定脸型/画风，完整效果必需。文件名须与 580 节点 widget 值一致，见作者原版 json。

拿到后放入 `models/loras/`，把服务器副本 json 里 580 节点恢复 `mode=0`，并把 `widgets_values[0]` 改成该文件名即可（其余 LoRA 保持 bypass 不动）。缺 Remix 时不要填 `none`，按第 4 节 bypass 580。

> 原版工作流里 lightx2v / DPO / slop 三个 LoRA 都在 bypass 的 557 节点中，**作者并未启用**，不要接。Uni3C 也是 bypass，controlnet 文件可选。

---

## 9. 故障速查

| 现象 | 原因 / 解决 |
|------|------|
| `comfy install` 卡死无进度 | 它内部 github clone，本机 github 不可达 → 弃用，改第 2 节 ghfast 拉 tar |
| `comfy run` 报 `workflow_unknown_nodes` (14 errors) | 原版 json 含虚拟节点 + 作者私有文件名 → 按第 4 节改 json（文件名 + 虚拟节点映射） |
| `comfy run` 报 `server_not_running` | ComfyUI 服务没起 → 先第 6 节启动，再 run |
| 节点报 `Invalid video/image file` | 输入文件名与 `input/` 里实际文件不符 → 改本地文件名去匹配工作流（不改成 json 则改文件名） |
| 某 custom node `Cannot import` / MISSING | 缺该节点依赖（accelerate/cv2 等）→ 装节点目录 requirements.txt 后重启 |
| `Can't import SageAttention` | 没装 sageattention → `pip install sageattention` |
| `Selected attention mode not available` | attention_mode 不是 sdpa/sageattn，或 sageattn 没装好 |
| `SafetensorError: incomplete metadata` | 模型文件损坏（同一最终文件被多个进程写入/截断）→ 删坏文件，用第 8 节分段 `.part` + 原子合并重下，下完 `safe_open` 校验 |
| 模型 URL 404 / `SCAIL-2/vae` 不存在 | SCAIL-2 仓只有 diffusion_models/loras → VAE/T5/CLIP 按第 8 节正确来源下载 |
| `double-valued cfg ... non-scaled fp8 model` | 用了 `umt5_xxl_fp8_e4m3fn_scaled.safetensors` → 换 `realung/umt5-xxl-enc-fp8_e4m3fn.safetensors` 的非 scaled 文件 |
| `unknown_enum_value` at 580 / `'none'` | LoRA 下拉没有 `none` → 缺 Remix 时 580 `mode=4` bypass，不填 `none` |
| 490 `SCAIL2ColoredMaskV2` CUDA OOM | `frame_load_cap=0` 吃进全部帧 → 479 改 161，490 `render_device='cpu'` |
| 执行中 WS 断开、ComfyUI 进程消失 | 922 帧整段 CPU/GPU 内存爆 → 同上限制帧数后重启服务 |
| 提交后进程**无声消失**（boot log 停在 VHS 加载、无 dmesg OOM） | 原生分辨率整段加载撞容器 cgroup 内存上限（wp08:15024 实测 cap=128G，`cat /sys/fs/cgroup/memory.max` 可查；2112×3840×922 帧 float32 ≈90G）被 SIGKILL。**内存不足的机器的特化手段**（不写进共享工作流）：471 `custom_width/height` 改 704/1280——生成尺寸由 531/532 INTConstant 硬编码（720→对 32 取整=704 × 1280），与加载尺寸无关，故输出等价；内存充足的机器保持 0 原生加载即可 |
| 抽帧后帧数比预期少一倍（如 10fps 只出 25 帧） | **force_rate 本身就会时域重采样**（`yieldable = total/src_fps × force_rate`），再叠 `select_every_nth` 就是双重抽帧。要 10fps 只设 `force_rate=10`、`select_every_nth` 保持 1 |
| `pkill -f "comfy run"` 后 SSH 会话 255 断开 | pkill 模式匹配到了自己这条 SSH 命令行（里面含同名字符串）→ 用 `pgrep -af "[c]omfy run"` 括号技巧排除自身 |
| 成片角色被拉伸/压扁 | 动作迁移模式下**画布 = 参考图**（实测：产物帧与参考图 SSIM≈0.37，与驱动视频帧 SSIM≈0.13）。参考图经 576 `_resize_bhwc`(crop=disabled) 拉伸进画布，比例不匹配即变形。**共享配置已改为 GetImageSize 自动跟随**（见第 4 节第 7 条），换参考图不用动手；若需手动指定，重新接线 531/532 INTConstant 即可（两节点保留未删）。驱动视频比例无需对齐 |
| 提交后 0.0x 秒 `success` 但没出新片 | 图哈希和上次完全相同命中缓存。改一个微小输入（如 steps）强制重算 |
| 几十秒"成功"跑完但没出片、GPU 全程空闲 | prompt 有节点校验失败被**静默跳过**（boot log 有 `Failed to validate prompt for output ... Output will be ignored`）→ 按 log 修对应节点；实测是 506 EmptyImage batch_size=0（见第 4 节） |
| `Value 0 smaller than min of 1: batch_size` | EmptyImage widgets 第 3 位是 batch_size（min=1）→ 506 改 `[512,8,1,0]` |
| `'sm_120' is not a recognized processor` / `LLVM ERROR: Cannot select`（5090） | sageattention 1.0.6 靠 triton，自带 triton 3.2 不认 Blackwell → `pip install -U triton`（≥3.3） |
| pip 装了 nvidia-cuda-nvcc-cu12 还是没有 nvcc（5090） | 该轮子只有 ptxas，缺 nvcc 本体 → 用 NVIDIA redist 的 cuda_nvcc tar 拼 CUDA_HOME（见第 2 节） |
| `comfy run --wait` 报 `ws_timeout` | 客户端 WebSocket 默认 120s 超时，**服务端照常跑完**；看 boot log 的 `Prompt executed in` 或 `--timeout` 调大，属装饰性报错 |
| `libGL.so.1: cannot open` | 缺 `libgl1` → `sudo apt install libgl1` |
| LayerStyle / VideoHelperSuite 导入失败 | 多半是 libGL 缺失，装 libgl1 后重启 |
| 后台下载进程随 SSH 断开死 | 普通 nohup 在 SSH 命令结束被杀 → 用 `setsid bash -c '...' & disown` 真正脱离 |
| github 直连 0.02MB/s / clone 失败 | github 不可达 → 用 `ghfast.top` / `gh-proxy.com` 镜像拉 tar，不直连 clone |

> 网络受限环境铁律：**ComfyUI 主程序用 ghfast 拉 tar（非 comfy install / 非 git clone）；节点同法；模型走 hf-mirror；任何直连 github 的 clone 都会死。**

---

## 10. 作者原版配置（复现用，服务器副本只按第 4 节做兼容修改）

> 以下为作者原版工作流的实际配置。Remix 由用户自备放入 `loras/`。
> **工作流 json 只在第 4 节列出的兼容项内改**；不改主链路结构/连线。

- 底模：`wan2.1_14B_SCAIL_2_fp8_scaled`（fp8_scaled 量化，base_precision=fp16，quantization=disabled，load_device=offload_device）
- attention_mode：`sageattn`（需 pip install sageattention）
- SamplerSettings steps：**6**（作者原版值）
- **LoRA：仅 Remix（580 节点，mode=0）**；557 里的 lightx2v/DPO/slop 保持 bypass（mode=4 / none）
- Uni3C：bypass（不加载）
- SAM3：启用（主链路遮罩/姿态）
- 分辨率/帧数：作者 json 内 576 节点保存的是 832×480 / 81 帧窗口；但 wp08 实跑 `ref.mp4` 时遮罩/输出链路按驱动视频生成 704×1280，479 限制为 161 帧后稳定。24G 机器不要 `frame_load_cap=0` 跑 922 帧整段。
- 输入：参考图 + 驱动视频（文件名匹配工作流写死的，不改 json 则改本地文件名去匹配）

---

## 11. 完整部署顺序速查（空服务器 → 出片）

```
1. pip install comfy-cli              # 装 comfy 命令（独立包）
2. 第 2 节：ghfast 拉 ComfyUI tar + 装依赖 + libgl/ffmpeg + sageattention
3. 第 3 节：ghfast 拉 8 个 custom nodes + 装节点依赖
4. 第 4 节：改 json（455/471 文件名 + 虚拟节点映射 + 576 BOOL + 无 Remix 时 580 bypass + 24G 帧数/490 CPU）
5. 第 8 节：setsid 后台 Range 分段下模型（hf-mirror；Remix 用户自备）
6. 第 5 节：传输入图/视频到 ~/comfy/ComfyUI/input/（文件名匹配）
7. 第 6 节：启动服务
8. 第 7 节：comfy run --workflow ~/scail_workflow.json --host 127.0.0.1 --port 8188 --wait
9. 从 ~/comfy/ComfyUI/output/ 取片（本轮成功产物在 literal `%date:...%` 目录下）
```

> 网络铁律：ComfyUI 主程序 / 节点用 **ghfast.top 拉 tar**（非 comfy install / 非 git clone）；模型走 **hf-mirror**；后台下载用 **setsid** 脱离 SSH。任何一步卡住参考第 9 节故障速查。

---

## 12. 本轮成功验证记录（wp08:19087，2026-07-19）

- 服务器副本：`~/scail_workflow.json`（本地原版 json 未改）；Remix 缺失，580 bypass；479 限 161 帧；490 `render_device='cpu'`。
- 命令：`comfy run --workflow ~/scail_workflow.json --host 127.0.0.1 --port 8188 --wait`
- 成功：`prompt_id=2e9dbc59-71bc-490a-8532-d04cb32d86bf`，耗时 `19:49`。
- 输出：`~/comfy/ComfyUI/output/%date:yyyy-MM-dd%/%date:yyyyMMdd_hhmmss%_Wanimate_00001-audio.mp4`，`704x1280`，`161` 帧，`60fps`，`2.684s`，含音频。
- 已拉回本地：`~/Downloads/scail2_20260719_171855_Wanimate_00001-audio.mp4`，SHA-256 `60deb955d9bd19ee07a3aa7e67ace772c829f9ea96e70fafa8a45082435547c2`。

---

## 13. 本轮成功验证记录（wp08:15024 / RTX 5090 32G，2026-07-23，**含 Remix**）

- 全新机从零部署：ComfyUI 0.28.0；torch 2.7.0+cu128 镜像预装（sm_120 matmul 直接可用）；nvcc 用 redist tar 12.8.93 拼 `~/cuda128`（pip 轮子缺 nvcc 本体）；sageattention 1.0.6 + **triton 3.7.1**（不升级 triton 则 sm_120 LLVM ERROR）。
- 系统内存 503G（远超 4090 机），922 帧整段的内存风险大减；显存峰值约 17.6G（161 帧）与 4090 持平。
- 服务器副本 `~/scail_workflow.json`：**580 mode=0 挂 Remix**（用户自备文件 scp 上传 3.68G，放 `models/loras/`，文件名与工作流一致）；490 `render_device='cpu'`；**506 改 `[512,8,1,0]`**（新版 EmptyImage schema，batch_size min=1）。
- 命令：`comfy run --workflow ~/scail_workflow.json --host 127.0.0.1 --port 8188 --wait --verbose`（客户端 `ws_timeout` 120s 报错属装饰，服务端正常完成）。
- 首轮 161 帧成功：`prompt_id=07bfd63c-f872-453e-9dc8-e558354ce01e`（506 batch_size 修复后），耗时 `16:15`（5090 vs 4090 的 19:49），输出 `704x1280`/`161` 帧/`60fps`/`2.684s`，已拉回 `~/Downloads/scail2_20260723_144000_Wanimate_00001-audio.mp4`。
- **帧数上限已放开（479=0）跑 922 帧整段**：原生 2112×3840 加载撞容器 128G cgroup 被杀（见第 9 节），本轮实际用 471=704/1280 加载变体提交（输出等价）。
- **共享工作流保持通用默认**：471 custom_width/height=0、479=0 已落盘本地仓库 `Scail-2+高阶工作流Plus+V4版【服务端兼容修改版】.json`；机器特化手段只记录在故障速查，不进共享配置（用户明确要求）。
- 时长说明：成片秒数 = 加载帧数 ÷ 输出fps；采样耗时约 6s/帧线性缩放，显存基本不随帧数涨。
- **10fps 快速变体**（同轮追加）：471 设 `force_rate=10`、`select_every_nth=1`（force_rate 自带时域重采样，别叠加 nth）→ 922→153 帧、15.3s、704×1280、含音频，耗时仅 `10:33`（vs 60fps 全帧 ~85min）。产物 `_Wanimate_00003-audio.mp4`（7.5MB），已拉回 `~/Downloads/scail2_10fps_20260723_155300_Wanimate_00003-audio.mp4`。该改为输出偏好，不进共享配置。

---

## 14. 本轮成功验证记录（wp08:13988 / RTX 4090 24G，2026-07-24，**hf-mirror 封 IP 绕行 + Remix**）

- 全新机从零部署，runbook 主流程不变。差异与替补链路：
  - **hf-mirror 突发按 IP 硬限速**：前 ~15GB 全速（数分钟内），随后所有连接（含单流小 Range、裸 TCP connect）SSL 握手超时，持续 3h+ 未恢复。与并发模式无关。
  - **底模 17.7G**：8 段 Range 下到 7/8 段后 part5 卡死 → 发现 **ModelScope 存在 `Comfy-Org/SCAIL-2` 镜像仓**（文件与 HF 逐字节一致，~13MB/s）→ 按 part5 的全局字节区间 `[11059116790, 13270940147]` Range 补齐（注意：本次脚本把 gend 误写 +40000，多出的尾部 truncate 即可，内容无损）→ 手动 cat 8 段合并 + safe_open 校验通过。
  - **CLIP（2.53G）**：改用 ModelScope `AI-ModelScope/CLIP-ViT-H-14-laion2B-s32B-b79K` 的 `model.safetensors`（3.94G fp32 **全集**，含 vision_model.* 键，ComfyUI clip_vision 只取视觉塔 = 与 2.53G 文件等效）→ 实跑验证通过。
  - **T5 fp8（6.7G）**：hf-mirror 磨到 6/8 段后速率跌至 17KB/s → 放弃，改从 ModelScope `Comfy-Org/Wan_2.1_ComfyUI_repackaged` 下 **`umt5_xxl_fp16.safetensors`（11.4G）重命名为 fp8 文件名**顶替。非 scaled、精度更高；289 节点 dtype widget 为 bf16，加载即转，**实跑验证通过**（副作用：显存峰值 17.6G→18.9G，encode 阶段略慢）。
  - **SAM3（1.7G，唯一 hf-mirror 独有）**：服务器任何路径都不通；**本地 Mac 系统代理 127.0.0.1:7890 可用**（urllib 走代理会 SSLEOFError，**curl -x 正常**，HF 直连 2.3MB/s）→ 本地下好后 scp 断点续传（`tail -c +N | ssh 'cat >>'`）+ sha256 校验。
  - facebook/sam3（ModelScope）**不能**顶替 multiplex：键名是 `detector_model.*`/`tracker_neck.*`（transformers 式），ComfyUI 识别要 `detector.*`+`tracker.*` 顶层键（见 model_detection.py:1053）。
- 服务器副本 `~/scail_workflow.json`：580 mode=0 挂 Remix（本地上传 3.68G）；479=161；490=cpu；其余随共享配置。
- 成功：`prompt_id=c8f69c61-2252-41e3-b34d-55acda204701`，耗时 `25:23`（fp16 T5 略拖慢 encode）。输出 `896x1184`（GetImageSize 自动跟随参考图 896×1200 → 对 32 取整）、`161` 帧、`60fps`、`2.683s`，含音频。
- 已拉回本地：`~/Downloads/scail2_20260724_172200_Wanimate_00001-audio.mp4`，SHA-256 `873ef34476e644ea72ef7cfdce31ad227a018faddb64b7f2aef3352a1b97d89e`。
- **教训**：再次踩 `pkill -f "python3 main\.py"` 自杀（远程 bash -c 命令串里含同名字面量）→ 一律 `pgrep -af "[m]ain.py"` 括号技巧。

---

## 15. 922 帧整段（15s 全时长）成功记录（wp08:13988 / 4090 24G，2026-07-25）

- 目标：`ref.mp4` 922 帧全时长（60fps 出 15.35s）。**显存是常量级**（分块采样，峰值 20.2G 与帧数无关）；**系统内存才是线性瓶颈**。
- 首跑失败：479=0 + 471=704/1280 配置下仍在 SAM3 追踪完成后被 SIGKILL（boot log 停在 490 警告处，dmesg 无记录）——加载尺寸疑似没真正压下去（dict 式 widgets 转换存疑），原生 2112×3840×922 float32 ≈90G 撞 cgroup 128G。
- **解法（有效）**：用 ffmpeg 把驱动视频**本体**降到 704×1280（`input/ref_704.mp4`，`-vf scale=704:1280 -crf 18`），471 指到新文件。输入侧操作，画布由参考图决定，**输出等价**。此后内存与 5090 那轮同水平。
- 成功：`prompt_id=adf0ca81-65dd-440b-a30c-3b0391322f9e`，耗时 `01:49:45`（13 chunk，SAM3 追踪 1:21，采样 ~6s/帧）。显存峰值 **20.22G**（fp16 T5 版，4090 24G 余量充足）；**cgroup 峰值 ~136.6G 贴线幸存**（帧张量全程驻留 RAM；后续任务前注意 `memory.current`，高就先重启清缓存）。
- 产物：`output/%date:...%_Wanimate_00002-audio.mp4`，896×1184、922 帧、60fps、15.35s、13.7MB，已拉回 `~/Downloads/scail2_full15s_20260724_195500_Wanimate_00002-audio.mp4`。
- 结论：**本机（4090 24G / cgroup 128G）能跑 15s 整段**，配置 = 479=0 + 输入视频本体≤704×1280 + 490=cpu；更长视频采样时间 ~6s/帧线性、显存不变、RAM 随帧数线性（当前 922 帧已到 cgroup 天花板，更长得先降加载分辨率）。
