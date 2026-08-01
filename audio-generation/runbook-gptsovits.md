# GPT-SoVITS — 部署与日文零样本推理 Runbook

> 目标机：wp08.unicorn.org.cn:28131（XDN1Lw）/ RTX 4090 24G / Ubuntu 22.04 裸容器（有 gcc/g++/make、wget/pip3；**无 cmake、无 ffmpeg、无 curl、无 git**）
> 目标：部署 RVC-Boss/**GPT-SoVITS**（v2ProPlus 权重），只用 Python API（`TTS_infer_pack`），不装 WebUI。
> 动机：IndexTTS2 仅支持中/英（日文假名全部 `<unk>`，见 runbook-indextts.md §5），日文文本的语音克隆走 GPT-SoVITS。
> 部署位置：`~/GPT-SoVITS`（uv 管 CPython 3.11 + `.venv`）；模型 `~/GPT-SoVITS/GPT_SoVITS/pretrained_models`（~1.4G）。
> 本 runbook 基于 2026-08-01 实测，全部下载在服务器上直接完成。

## 0. 结论速览

- v2ProPlus 日文零样本验证通过：**无参考文本模式**（`prompt_text=""`）可用，零样本音色 + 日文文本出声正常（§4）。
- 网络铁律沿用本仓库惯例并新增两条：**github 直连死 → ghfast 拉 tar**（但 pyopenjtalk 的 open_jtalk 词典走 github r9y9 release **实测直连成功**，疑似该 release CDN 未被墙）；**pypi 走阿里源**；`download.pytorch.org/whl/cu128` 直连可达；**模型走 hf-mirror**（`lj1995/GPT-SoVITS`，HF API 会 308 到 huggingface.co，但 `/resolve/` 文件下载正常代理）；`dl.fbaipublicfiles.com`（fasttext 语言检测模型）直连可达。
- **torch 必须钉 2.8.0**：装最新（2.11）会让 `torchaudio.load` 走 torchcodec → 需要系统 ffmpeg 共享库（容器没有）→ 全线崩（§5）。
- GPT-SoVITS 是 **TTS**（prompt 音色 + 文字 → 语音），不是语音转换；不能把现成音频的韵律/表演直接迁移到新音色（那是 RVC/seed-VC）。

## 1. 网络摸底（2026-08-01 实测）

| 源 | 结果 | 用途 |
|---|---|---|
| github.com 直连 | ❌（codeload 死）/ ⚠️ r9y9 release 直连成功 | 代码 tar 走 ghfast；open_jtalk 词典直连 |
| `ghfast.top` | ✅ | 代码 tar、uv 的 CPython 镜像 |
| mirrors.aliyun.com/pypi | ✅ 快 | 全部 pypi 包（含 cmake wheel） |
| download.pytorch.org/whl/cu128 | ✅ 快 | torch/torchaudio 2.8.0 cu128 |
| hf-mirror.com | ✅（1.5~12MB/s 波动） | 全部预训练模型 |
| dl.fbaipublicfiles.com | ✅ | fasttext `lid.176.bin`（fast_langdetect 自动下） |

> 该容器 sshd 踢空闲会话（`Connection closed by 198.18.0.25`）。**长任务必须 `setsid ... & disown`**，日志落 /tmp 轮询；交互加 `ServerAliveInterval=15`。

## 2. 部署步骤（全新容器 → 可用）

### 2.1 装 uv + 拉代码（同 indextts runbook §2.1/2.2）

```bash
pip3 install -U uv -i https://mirrors.aliyun.com/pypi/simple/ --retries 20 --timeout 60
export PATH=$HOME/.local/bin:$PATH   # 后续每条 ssh 都要带
cd ~ && wget -q --tries=3 "https://ghfast.top/https://github.com/RVC-Boss/GPT-SoVITS/archive/refs/heads/main.tar.gz" -O gpt-sovits.tar.gz
gzip -t gpt-sovits.tar.gz && tar xzf gpt-sovits.tar.gz && mv GPT-SoVITS-main GPT-SoVITS && rm gpt-sovits.tar.gz
```

- 仓库**不含 LFS**：`GPT_SoVITS/pretrained_models/` 是空目录，模型单独下（§2.4）。

### 2.2 建 venv + 编译工具（关键：先装 cmake）

```bash
cd ~/GPT-SoVITS
export UV_PYTHON_INSTALL_MIRROR="https://ghfast.top/https://github.com/astral-sh/python-build-standalone/releases/download"
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python cmake imageio-ffmpeg --default-index "https://mirrors.aliyun.com/pypi/simple/"
# 静态 ffmpeg 二进制兜底（torchaudio 2.8 用不到，webui/tools 会用）
ln -sf ~/GPT-SoVITS/.venv/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2 ~/.local/bin/ffmpeg
```

- `cmake` 用 pypi wheel 即可（manylinux 二进制），不用 apt；pyopenjtalk/opencc 编译依赖它 + 系统 g++。

### 2.3 装 torch（钉 2.8.0）+ requirements

```bash
setsid bash -c 'cd ~/GPT-SoVITS && export PATH=$HOME/.local/bin:$PATH && {
  uv pip install --python .venv/bin/python "torch==2.8.0" "torchaudio==2.8.0" --index-url "https://download.pytorch.org/whl/cu128" &&
  uv pip install --python .venv/bin/python -r requirements.txt --default-index "https://mirrors.aliyun.com/pypi/simple/"
} > /tmp/gsv_env.log 2>&1; echo EXIT=$? >> /tmp/gsv_env.log' </dev/null & disown
```

- **不要装最新 torch**（实测 2.11.0+cu128 翻车）：torchaudio ≥2.9 的 `load()` 改为 torchcodec 实现，torchcodec 要系统 ffmpeg 共享库（libavcodec.so 等），裸容器没有 → `OSError: libtorchcodec_core4.so`。2.8.0 走 soundfile 后端，wav 加载零依赖。
- requirements 含 `pyopenjtalk>=0.4.1`（日文 g2p，源码编译 ~5min，需 §2.2 的 cmake）与 `--no-binary=opencc`（同样编译）；首次 `import pyopenjtalk` 自动下 open_jtalk 词典（github r9y9 release，22.6MB，实测直连成功）。
- `funasr`/`gradio<5` 等照常装（推理用不到但不碍事）；`extra-req.txt`（faster-whisper）是 ASR 打标用的，推理不用装。
- 实测全程 EXIT=0 约 12min；`.venv` 8.7G。

### 2.4 下模型（hf-mirror，wget -c 可断点续传）

```bash
setsid bash -c 'cd ~/GPT-SoVITS/GPT_SoVITS/pretrained_models && mkdir -p chinese-hubert-base chinese-roberta-wwm-ext-large v2Pro sv fast_langdetect && B="https://hf-mirror.com/lj1995/GPT-SoVITS/resolve/main" && {
  wget -q -c "$B/chinese-hubert-base/config.json"                -O chinese-hubert-base/config.json &&
  wget -q -c "$B/chinese-hubert-base/preprocessor_config.json"   -O chinese-hubert-base/preprocessor_config.json &&
  wget -q -c "$B/chinese-hubert-base/pytorch_model.bin"          -O chinese-hubert-base/pytorch_model.bin &&
  wget -q -c "$B/chinese-roberta-wwm-ext-large/config.json"      -O chinese-roberta-wwm-ext-large/config.json &&
  wget -q -c "$B/chinese-roberta-wwm-ext-large/tokenizer.json"   -O chinese-roberta-wwm-ext-large/tokenizer.json &&
  wget -q -c "$B/chinese-roberta-wwm-ext-large/pytorch_model.bin" -O chinese-roberta-wwm-ext-large/pytorch_model.bin &&
  wget -q -c "$B/s1v3.ckpt"                                      -O s1v3.ckpt &&
  wget -q -c "$B/v2Pro/s2Gv2ProPlus.pth"                         -O v2Pro/s2Gv2ProPlus.pth &&
  wget -q -c "$B/sv/pretrained_eres2netv2w24s4ep4.ckpt"          -O sv/pretrained_eres2netv2w24s4ep4.ckpt
} > /tmp/gsv_models.log 2>&1; echo EXIT=$? >> /tmp/gsv_models.log' </dev/null & disown
```

推理必需文件（v2ProPlus，总 ~1.2G）与官方尺寸（可用于校验完整性）：

| 文件 | 尺寸 (bytes) | 说明 |
|---|---:|---|
| `chinese-hubert-base/pytorch_model.bin` | 188,811,417 | SSL 特征（全日文也用中文 hubert，官方如此） |
| `chinese-roberta-wwm-ext-large/pytorch_model.bin` | 651,225,145 | 文本 BERT |
| `s1v3.ckpt` | 155,284,856 | GPT/t2s 权重（v3/v4/v2Pro 系共用） |
| `v2Pro/s2Gv2ProPlus.pth` | 200,125,741 | SoVITS 生成器（**s2D 是判别器，训练才用，不用下**） |
| `sv/pretrained_eres2netv2w24s4ep4.ckpt` | 107,528,697 | 说话人编码器（v2Pro/+ 用） |

- 选 v2ProPlus 而非 v4：README 明确 v2ProPlus 性能超 v4；v4 原生 48k 输出是另一卖点，需要可另行下载 `gsv-v4-pretrained/{s2Gv4.pth,vocoder.pth}`。
- **日文不需要 G2PWModel**（仅中文 TTS 用）；UVR5 人声分离模型只有训练前处理才用。
- `fast_langdetect` 目录**必须手动 mkdir**（§5），`lid.176.bin`（126MB）首次推理自动从 dl.fbaipublicfiles.com 下载。

### 2.5 推理配置（自定义 yaml，绕开默认 cpu 配置）

`~/GPT-SoVITS/GPT_SoVITS/configs/tts_infer_v2proplus.yaml`：

```yaml
custom:
  bert_base_path: GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
  cnhuhbert_base_path: GPT_SoVITS/pretrained_models/chinese-hubert-base
  device: cuda
  is_half: true
  t2s_weights_path: GPT_SoVITS/pretrained_models/s1v3.ckpt
  version: v2ProPlus
  vits_weights_path: GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth
```

> 仓库自带 `tts_infer.yaml` 里 v2ProPlus 默认 `device: cpu, is_half: false`；`custom` 段是 cuda+half 版。

### 2.6 验证（Python API，无参考文本 + 日文）

服务器上留有脚本 `~/GPT-SoVITS/gen_gsv.py`：

```python
import os, sys, time
now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(now_dir); sys.path.append(os.path.join(now_dir, "GPT_SoVITS"))  # 关键：AR 包在子目录
import soundfile as sf
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

tts = TTS(TTS_Config("GPT_SoVITS/configs/tts_infer_v2proplus.yaml"))
req = {
    "text": open("custom/text1.txt", encoding="utf-8").read().strip(),
    "text_lang": "ja",
    "ref_audio_path": "prompt.wav",       # 3~10s 干净高能量人声切片
    "prompt_text": "",                    # 无参考文本模式（零样本）
    "prompt_lang": "ja",
    "top_k": 15, "top_p": 1.0, "temperature": 1.0,
    "text_split_method": "cut5",          # 按标点切句
    "batch_size": 4, "batch_threshold": 0.75, "split_bucket": True,
    "speed_factor": 1.0, "fragment_interval": 0.3,
    "seed": -1, "parallel_infer": True, "repetition_penalty": 1.35,
    "return_fragment": False, "streaming_mode": False,
}
sr, audio = None, None
for sr_, audio_ in tts.run(req):
    sr, audio = sr_, audio_
sf.write("outputs/out1.wav", audio, sr)   # audio 为 int16 ndarray
```

运行：`cd ~/GPT-SoVITS && .venv/bin/python gen_gsv.py`（首次推理会自动下 `lid.176.bin`）。
判别：输出 32000Hz 单声道；peak 0.5~1.0（int16 归一化后）；日文文本可懂。

## 3. Python API 用法要点（日常）

- **prompt 音频**：3~10s 干净人声，**情绪/韵律会从 prompt 迁移到输出**。从长音频选高能量段：`ffmpeg -i long.wav -af "asetnsamples=88200,astats=metadata=1:reset=0,ametadata=mode=print:key=lavfi.astats.Overall.RMS_level:file=-" -f null -` 找 RMS 高的 2s 窗，再 `ffmpeg -ss <t> -to <t+8> -i long.wav -ac 1 -c:a pcm_s16le prompt.wav`。
- **无参考文本模式**：`prompt_text=""`（v2+ 支持，零样本）；有 prompt 转写文本时填上会略稳。
- 语言标记：`text_lang`/`prompt_lang` 支持 `zh/ja/en/ko/yue`（中英日韩粤）；`split-lang` + fasttext 自动分句检测也依赖 `lid.176.bin`。
- `tts.run(req)` 是生成器，非分段模式只 yield 一次 `(sr, int16 ndarray)`。
- 输出采样率随版本：v2ProPlus 原生 **32000Hz**（v3=24k，v4=48k）。
- 长文本用 `cut5`（按标点）；`parallel_infer + batch_size=4 + split_bucket` 提速明显（471 字日文 warm 22.5s 出 121s 音频）。
- 复现：固定 `seed`（默认 -1 随机）。
- 超分模型（48k 化）与 UVR5 未部署，见 README `tools/AP_BWE_main`。

## 4. 验证记录（wp08:28131 / XDN1Lw / 4090 24G，2026-08-01）

| 项 | 值 |
|---|---|
| 环境 | uv 0.12.1 / CPython 3.11 / **torch 2.8.0+cu128**（2.11 翻车降级，见 §5）/ pyopenjtalk 0.4.1 源码编译 |
| 模型 | hf-mirror `lj1995/GPT-SoVITS`：v2ProPlus 五件套 ~1.2G（wget -c，~10min）+ lid.176.bin 126M 自动 |
| prompt | 145s 参考音源的前段高能量 8s 切片（t=2~10s，mono），无参考文本 |
| 日文合成 ① | text1.txt（242字）→ 69.54s @32000Hz，peak=0.798；推理 82.1s（含 CUDA warmup+首句编译） |
| 日文合成 ② | text2.txt（471字）→ 121.04s @32000Hz，peak=0.741；推理 **22.5s（RTF 0.19，warm）** |
| 产物 | `~/GPT-SoVITS/outputs/out{1,2}.wav`，已拉回本地（ffprobe 验证 pcm_s16le 完整） |
| 磁盘 | .venv 8.7G + pretrained_models 1.4G |

对比 IndexTTS2 同文本：GPT-SoVITS 节奏更慢（121s vs 110s @ text2）；IndexTTS2 日文直接不可用。

## 5. 故障速查

| 症状 | 原因 / 处理 |
|---|---|
| `ModuleNotFoundError: No module named 'AR'` | `sys.path` 只加了仓库根；**必须同时 append `GPT_SoVITS/` 子目录**（仿 api_v2.py L110-111） |
| `OSError: Could not load this library: libtorchcodec_core4.so` | 装了最新 torch（2.11）→ torchaudio≥2.9 `load()` 走 torchcodec，要系统 ffmpeg 共享库。处理：`uv pip install "torch==2.8.0" "torchaudio==2.8.0" --index-url .../cu128` 降级（已验证）；备选 apt 装 ffmpeg 系库（未实测） |
| `FileNotFoundError: fast-langdetect: Cache directory not found` | 包要求 cache 目录预先存在：`mkdir -p GPT_SoVITS/pretrained_models/fast_langdetect`，之后自动下 lid.176.bin |
| `pyopenjtalk` 编译失败 | 缺 cmake → `uv pip install cmake`（pypi wheel）；g++ 容器已有 |
| pyopenjtalk 词典下载卡死 | open_jtalk dic 走 github r9y9 release（实测直连成功）；若失效，手动 `ghfast.top/https://github.com/r9y9/open_jtalk/releases/download/v1.11.1/open_jtalk_dic_utf_8-1.11.tar.gz` 解到 site-packages/pyopenjtalk/dic |
| hf-mirror API 308 跳转 huggingface.co | 正常现象：API 不代理，**`/resolve/main/...` 文件下载正常**；别用 `huggingface-cli download`（走 API），用 wget 直链 |
| 输出音色不像/情绪平 | prompt 段没选好：换 3~10s 内情绪更饱满的切片；情绪随 prompt 迁移，不与文本情绪自动对齐 |
| 想"换音色保表演"（非 TTS） | GPT-SoVITS 做不到，属语音转换：看 RVC / seed-VC |
| IndexTTS2 日文乱码 | 预期行为：IndexTTS2 仅中英，假名 BPE 全 `<unk>` → 用本 runbook 的 GPT-SoVITS |
