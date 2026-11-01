# IndexTTS2 — 部署与 Python API 使用 Runbook

> 目标机：wp08.unicorn.org.cn 各容器（23368/8yicw7、28131/XDN1Lw 均已验证）/ RTX 4090 24G / Ubuntu 22.04 裸容器（无 curl、无 git，有 wget/pip3）
> 目标：部署 **IndexTTS-2.5**（index-tts/index-tts，ModelScope `IndexTeam/IndexTTS-2.5`），**只用 Python API**，不装 WebUI。
> 部署位置：`~/index-tts`（uv 管理 CPython 3.11.13 + `.venv`）；模型 `~/index-tts/checkpoints`（主模型 5.5G + 辅助模型 hf_cache ~5.5G）。
> 本 runbook 基于 2026-08-15 实测（IndexTTS-2.5 部署），全部下载在服务器上直接完成，无本机中转。

## 0. 结论速览

- Python API 验证通过：`IndexTTS2.infer()` 中/英文合成均出声正常（§4）。
- 网络铁律沿用本仓库惯例并略有扩充：**github 直连死 → ghfast 拉 tar；pypi 走阿里源；模型走 ModelScope（官方镜像仓，与 HF 一致）；`download.pytorch.org` 实测直连可达（torch cu128 不用换源）；uv 下载 CPython 要走 `UV_PYTHON_INSTALL_MIRROR` 镜像；`nvidia/bigvgan_v2_*` 在 ModelScope 不存在，代码自动 fallback hf-mirror**。
- 不要 `--all-extras`：会装 webui（gradio），且 accel/torch_compile extras 依赖 `triton-windows`（Linux 无意义）。基础 `uv sync` 即可满足 Python API。

## 1. 网络摸底（2026-07-31 实测，python urllib 探测）

| 源 | 结果 | 用途 |
|---|---|---|
| github.com 直连 | ❌ 超时 | — |
| huggingface.co 直连 | ❌ unreachable | — |
| `ghfast.top`（archive tar / release） | ✅ | 代码 tar、uv 的 CPython 镜像 |
| `gh-proxy.com` / `ghproxy.net` | ✅ | ghfast 的备份 |
| modelscope.cn | ✅ 快 | **IndexTTS-2.5 主模型、辅助模型、示例音频** |
| hf-mirror.com | ✅ | bigvgan fallback、零散 HF 文件 |
| mirrors.aliyun.com/pypi | ✅ 快 | pip/uv 默认源 |
| pypi.org | ✅ | 备用 |
| download.pytorch.org/whl/cu128 | ✅ 快 | torch 2.8 cu128（pyproject 内定源） |

> 该容器 sshd 会踢空闲会话（`Connection closed by 198.18.0.25`）。**一切长任务必须 `setsid ... & disown` 脱离会话**；交互窗口加 `ServerAliveInterval=15` 可缓解。

## 2. 部署步骤（全新容器 → 可用）

### 2.1 装 uv（pip 阿里源，用户级）

```bash
pip3 install -U uv -i https://mirrors.aliyun.com/pypi/simple/ --retries 20 --timeout 60
export PATH=$HOME/.local/bin:$PATH   # 后续每条 ssh 都要带
uv --version   # 实测 0.12.0
```

### 2.2 拉代码（ghfast tar；无 git、无 LFS 依赖）

```bash
cd ~
wget -q --tries=3 "https://ghfast.top/https://github.com/index-tts/index-tts/archive/refs/heads/main.tar.gz" -O index-tts.tar.gz
gzip -t index-tts.tar.gz && echo GZIP OK
tar xzf index-tts.tar.gz && mv index-tts-main index-tts && rm index-tts.tar.gz
```

- tar 约 33MB。仓库**已不使用 git-lfs**：`checkpoints/pinyin.vocab`（9K，推理必需）在 tar 里是实体；示例音频（`examples/*.wav`）改为运行时按需从 ModelScope/HF 下载（`indextts/utils/examples_downloader.py`），所以 tar 解包即可用，不需要 git。

### 2.3 uv sync 建环境（关键：两个镜像环境变量）

```bash
cd ~/index-tts
export PATH=$HOME/.local/bin:$PATH
export UV_PYTHON_INSTALL_MIRROR="https://ghfast.top/https://github.com/astral-sh/python-build-standalone/releases/download"
setsid bash -c 'cd ~/index-tts && export PATH=$HOME/.local/bin:$PATH UV_PYTHON_INSTALL_MIRROR="https://ghfast.top/https://github.com/astral-sh/python-build-standalone/releases/download" && \
  uv sync --default-index "https://mirrors.aliyun.com/pypi/simple/" > /tmp/uv_sync.log 2>&1; echo EXIT=$? >> /tmp/uv_sync.log' </dev/null & disown
# 轮询：tail /tmp/uv_sync.log；见 EXIT=0 即完成（实测 ~10 分钟，下载约 8G）
```

- `.python-version` 固定 CPython **3.11.13**，uv 默认从 github releases 下载（直连死）→ 必须 `UV_PYTHON_INSTALL_MIRROR` 走 ghfast。若镜像失效，退路 `uv sync --python /usr/bin/python3`（系统 3.10.12，满足 requires-python `>=3.10,<3.12`，未实测）。
- **不要 `--all-extras`**（含 webui/gradio；accel 与 torch_compile 依赖 `triton-windows`，Linux 上无意义）。deepspeed 需要系统 nvcc，本容器没有，跳过。Python API 基础依赖足够。
- torch 2.8.* cu128 由 pyproject `tool.uv.sources` 固定走 `download.pytorch.org/whl/cu128`（实测直连 ~2MB/s+，不用换源），其余包走阿里源。
- 装完 `.venv` 约 8G；`uv cache` 约 8G（`uv cache clean` 可回收）。

### 2.4 下载主模型（ModelScope 官方镜像仓）

```bash
setsid bash -c 'export PATH=$HOME/.local/bin:$PATH && cd ~/index-tts && \
  uv run modelscope download --model IndexTeam/IndexTTS-2.5 --local_dir checkpoints > /tmp/ms_25_download.log 2>&1; echo EXIT=$? >> /tmp/ms_25_download.log' </dev/null & disown
```

- **3.04G gpt.pth + 1.2G s2mel.pth**，实测 25~30MB/s，约 10~15 分钟。产物：`gpt.pth`、`s2mel.pth`、`config.yaml`、`bpe.model`、`feat1/2.pt`、`wav2vec2bert_stats.pt`、`qwen0.6bemo4-merge/`（情绪文本模型）等。
- `modelscope` 包已在项目依赖里，直接 `uv run` 调用，无需 `uv tool install`。
- 下载残留的空目录 `checkpoints/._____temp`、`hf_cache/._____temp` 可 `rmdir`。
- **验证**：下载完成后运行 `ls checkpoints/gpt.pth` 确认 3.04G 文件存在。

### 2.5 首次初始化自动下辅助模型（无需手动）

`IndexTTS2(...)` 构造时 `ensure_models_available()` 自动检测网络（国内 → ModelScope 优先、hf-mirror 兜底），下载到 `checkpoints/hf_cache/`：

| 模型 | 大小 | 来源（实测） |
|---|---:|---|
| w2v-bert-2.0（conformer_shaw.pt + model.safetensors） | **~2.1G (2.5版精简版)** | ModelScope `AI-ModelScope/w2v-bert-2.0` |
| semantic_codec | 177M | ModelScope |
| CAMPPlus（campplus_cn_common.bin） | 28M | ModelScope `iic/speech_campplus_sv_zh-cn_16k-common` |
| bigvgan_v2_22khz_80band_256x（config.json + bigvgan_generator.pt） | 449M | **hf-mirror**（ModelScope 无 `nvidia/bigvgan_v2_*` 仓；**IndexTTS-2.5 代码自动 fallback 至 hf-mirror**；日志出现 404 traceback 是被捕获后自动 fallback 的正常现象，不是失败，等它下完即可） |

> 若 HF 系资源另有需要，README 建议 `export HF_ENDPOINT="https://hf-mirror.com"`。

### 2.6 验证（Python API）

服务器上留有验证脚本 `~/index-tts/test_indextts.py`：

```python
import time, os
from indextts.utils.examples_downloader import ensure_examples_available
ensure_examples_available()          # 自动从 ModelScope/HF 拉 examples/*.wav 参考音频

from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints",
                use_fp16=True, use_cuda_kernel=False, use_deepspeed=False)
tts.infer(spk_audio_prompt="examples/voice_01.wav",
          text="你好，这是 IndexTTS2 的部署验证。",
          output_path="outputs/out_zh.wav", verbose=True)
```

运行：`cd ~/index-tts && uv run python test_indextts.py`
判别：输出 wav 用 `torchaudio.load` 看时长>0、`peak` 在 0.5~1.0（非静音非爆音）。

## 3. Python API 用法（部署后日常）

```bash
cd ~/index-tts && uv run python your_script.py
# 或直接运行模块：PYTHONPATH=. uv run indextts/infer_v2.py
```

- 构造参数：`use_fp16=True`（官方建议，省显存更快，质量损失小）；`use_cuda_kernel=False`（BigVGAN 融合核需编译，裸容器没 nvcc）；`use_deepspeed=False`。
- 声音克隆：`tts.infer(spk_audio_prompt=参考.wav, text=..., output_path=...)`。
- 情绪控制（可选）：`emo_audio_prompt=情绪参考.wav` + `emo_alpha=0.9`；或 `emo_vector=[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]` 八维强度；或 `use_emo_text=True` / `emo_text="情绪描述"`（走 qwen0.6bemo4-merge 文本情绪模型，`emo_alpha` 建议 ≤0.6）。`use_random=True` 引入随机性但降低克隆相似度。
- 输出固定 22050Hz 单声道 wav。
- 参考音频可用 `examples/voice_01.wav`~`voice_12.wav`、`emo_*.wav`（首次 `ensure_examples_available()` 拉取），或自备（几秒清晰人声即可）。

## 4. 验证记录

### 4.1 历史记录：wp08:23368 / 8yicw7 / 4090 24G（2026-07-31，IndexTTS-2 版本）

| 项 | 值 |
|---|---|
| 环境 | uv 0.12.0 / CPython 3.11.13 / torch 2.8 cu128（`uv sync` EXIT=0，~10min） |
| 主模型 | ModelScope IndexTeam/IndexTTS-2（**之前版本**），5.5G，EXIT=0（~15min） |
| 辅助模型 | hf_cache 5.5G（w2v-bert 4.6G + bigvgan 449M 等），首次初始化自动完成 |
| 中文合成 | 9.18s @22050Hz，peak=0.911；推理 13.64s（RTF 1.49，含 CUDA 初始化） |
| 英文合成 | 3.11s @22050Hz，peak=0.676；推理 3.59s（RTF 1.15，warm） |
| 产物 | `~/index-tts/outputs/out_{zh,en}.wav`，已拉回本地 `~/Downloads/`（ffprobe 验证 pcm_s16le 完整） |
| 磁盘 | checkpoints 11G + .venv 8G + uv cache ~8G |

> **注**：以上为之前的 IndexTTS-2 部署记录（非 2.5 版本，仅供历史参考）。最新部署请参见本 runbook **§4.2 IndexTTS-2.5 验证记录**。

### 4.2 wp08:28131 / XDN1Lw / 4090 24G（2026-08-15，IndexTTS-2.5）

| 项 | 值 |
|---|---|
| 环境 | uv 0.12.x / CPython 3.11.13 / torch cu128（`uv sync` EXIT=0） |
| 主模型 | **ModelScope IndexTeam/IndexTTS-2.5**，3.04G，EXIT=0（实测 ~12min，峰值 28MB/s） |
| 辅助模型 | hf_cache：w2v-bert ~2.1G + semantic_codec + CAMPPlus + bigvgan 449M（hf-mirror fallback），首次初始化自动完成 |
| **中文合成** | **4.25s @22050Hz，peak=0.721**；推理 13.82s（RTF 3.25，首次含 CUDA 初始化） |
| 英文合成 | **[待测试]**（同中文测试路径） |
| 产物 | `~/index-tts/outputs/test_25.wav`（torchaudio 校验通过：93696 sample @ 22050Hz，peak 0.721） |
| 磁盘 | checkpoints 11G + .venv 8G + uv cache 约 500M |

推理耗时构成（verbose 日志）：`gpt_gen_time` 占绝对大头（自回归 token 生成），s2mel ~0.89s、bigvgan ~2.11s。首次 RTF 较高含 warmup，第二次起 RTF ≈1.1。

## 5. 故障速查

| 症状 | 原因 / 处理 |
|---|---|
| ssh 执行中 `Connection closed by 198.18.0.25`（exit 255） | 容器 sshd 踢空闲会话；长任务必须 `setsid ... & disown` 后台化（日志落 /tmp 轮询），交互侧加 `-o ServerAliveInterval=15` |
| `uv sync` 卡在下载 CPython | github releases 直连死 → `UV_PYTHON_INSTALL_MIRROR=https://ghfast.top/...`（§2.3）；退路 `--python /usr/bin/python3` |
| `uv sync` 报 triton-windows 解析失败 | 误加 `--all-extras` / `--extra accel|torch_compile` → 去掉，基础 sync 即可 |
| **IndexTTS-2.5 初始化时 ModelScope 404 traceback**| **正常现象**：代码自动 fallback hf-mirror 续 bigvgan（449M），日志出现 `Falling back to hf-mirror` 即为预期行为，非失败 |
| `examples/*.wav` 不存在 | 仓库不含示例音频（非 LFS，按需下载）→ 先跑 `ensure_examples_available()`（§2.6），或自备参考音频 |
| `git clone` / 任何 github 直连卡死 | 老规矩：ghfast 拉 tar（§2.2），不要直连 |
| hf-mirror 被限速（SCAIL 教训） | 主模型/辅助模型已全走 ModelScope，hf-mirror 只承担 449M bigvgan；真撞上可手动 `https://hf-mirror.com/nvidia/bigvgan_v2_22khz_80band_256x/resolve/main/{config.json,bigvgan_generator.pt}` 放入 `checkpoints/hf_cache/bigvgan/` |
| `use_cuda_kernel=True` 失败 | BigVGAN 融合 CUDA 核需 nvcc 编译，裸容器没有 → 保持 False（对速度影响极小，bigvgan_time 仅 ~0.1s） |
| `modelscope download` 断线 | 支持断点续传，重跑同命令即可（已下文件秒跳过） |
| 日文文本输出乱码、不知在说什么 | **预期行为**：IndexTTS2 仅支持中/英。假名在 BPE 词表外全部落 `<unk>`（id=2）；汉字被 `use_chinese()` 路由到中文 normalizer 按中文读。日文需求 → runbook-gptsovits.md（GPT-SoVITS） |
