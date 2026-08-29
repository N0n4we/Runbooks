#!/usr/bin/env python3
"""Run the FeiHou MiniMax H3 workflow on the configured RTX 5090.

The workflow is kept in ComfyUI's exported UI format.  Before execution it is
converted by ``comfy run --print-prompt`` on the target, then the API graph is
patched so that model filenames, prompt, media slots, seed and output prefix
are deterministic.  The single Remix checkpoint is assigned to both FL2VA and
REF2VA; the supplied media flags select which path is executed.

### 1. 文生视频 / FL2VA

python3 video-generation/h3-feihou/run-h3.py \
  --prompt "A person walks slowly through a sunny garden." \
  --seconds 5 \
  --seed 42

### 2. 首尾帧 FL2VA

python3 video-generation/h3-feihou/run-h3.py \
  --prompt "The subject moves naturally." \
  --image first.png \
  --last-image last.png \
  --seconds 5 \
  --seed 42

### 3. REF2VA 混合参考

python3 video-generation/h3-feihou/run-h3.py \
  --prompt "The person in <Picture 1> follows the motion in <Video 1>." \
  --ref-image person.png \
  --ref-video motion.mp4 \
  --ref-audio ambience.wav \
  --seconds 5

可重复传入多个 --ref-image、--ref-video、--ref-audio。

### 常用参数

--resolution 480P
--aspect 16:9
--lora-strength 0.75
--second-sampling
--out ~/Downloads/result.mp4
--no-download

脚本会自动连接服务器、启动 ComfyUI、上传输入、执行推理并下载结果。
--image/--last-image 自动使用 FL2VA；--ref-* 自动使用 REF2VA。两类输入不能在同一次
运行中混用。
"""

from __future__ import annotations

import argparse
import json
import random
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path


SERVER = "root@connect.westb.seetacloud.com"
SSH_PORT = 27454
REMOTE_COMFY = "~/comfy/ComfyUI"
REMOTE_INPUT = "~/comfy/ComfyUI/input"
REMOTE_OUTPUT = "~/comfy/ComfyUI/output"
REMOTE_VENV = "~/h3-venv"
WORKFLOW = Path(__file__).with_name("MiniMaxH3-FeiHou-Easy-H3.json")

REMIX_MODEL = "FeiHou_MiniMax-H3_Remix_v0.6_int8_convrot_v2.safetensors"
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_int8_convrot_uncensored.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
TURBO_LORA = "H3/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"

SSH_OPTS = [
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ConnectTimeout=15",
    "-o", "ServerAliveInterval=15",
]


def local_run(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
    return subprocess.run(cmd, check=True, text=True, capture_output=capture)


def ssh(command: str, *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", *SSH_OPTS, "-p", str(SSH_PORT), SERVER, command],
        check=True,
        text=True,
        capture_output=capture,
    )


def scp_to(local: Path, remote: str) -> None:
    local_run(["scp", *SSH_OPTS, "-P", str(SSH_PORT), str(local), f"{SERVER}:{remote}"])


def snap_frames(seconds: float, fps: int = 24) -> int:
    """H3 uses the 17k+5 frame grid (5, 22, 39, ...)."""
    target = max(5, round(seconds * fps))
    return max(5, ((target - 5 + 16) // 17) * 17 + 5)


def parse_size(value: str) -> tuple[int, int]:
    try:
        width, height = (int(x) for x in value.lower().split("x", 1))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("尺寸必须是 WxH，例如 1344x768") from exc
    if width < 32 or height < 32 or width % 32 or height % 32:
        raise argparse.ArgumentTypeError("宽高必须是不小于 32 的 32 倍数")
    return width, height


def infer_mode(args: argparse.Namespace) -> None:
    """Select the path from supplied media; there is no user-facing mode flag."""
    fl2va_inputs = bool(args.image or args.last_image)
    ref2va_inputs = bool(args.ref_image or args.ref_video or args.ref_audio)
    if fl2va_inputs and ref2va_inputs:
        raise SystemExit(
            "当前 FeiHou 工作流不支持同一次采样混合 FL2VA 首/末帧与 REF2VA 参考媒体；"
            "请拆成两次运行。REF2VA 内部可以混合图片、视频和音频。"
        )
    args.mode = "ref2va" if ref2va_inputs else "fl2va"


def _direct_link(value: object) -> tuple[str, int] | None:
    """Return an API-graph input link, if ``value`` is one."""
    if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
        return value[0], int(value[1])
    return None


def prune_second_sampling(prompt: dict[str, dict]) -> None:
    """Remove the optional second-pass subgraph from the v2 workflow.

    The exported v2 workflow contains both samplers.  Merely setting the
    loader's second model to ``none`` is insufficient: ComfyUI would still
    execute the second sampler with a null MODEL.  This removes only nodes
    exclusive to that branch while retaining shared first-pass decodes.
    """
    h3_nodes = [n for n in prompt.values() if n.get("class_type") == "FeiHouEasyH3"]
    if len(h3_nodes) != 1:
        return
    h3_id = next(node_id for node_id, node in prompt.items() if node is h3_nodes[0])
    second_attention = next(
        (node_id for node_id, node in prompt.items()
         if node.get("class_type") == "ModelAttentionBackend"
         and _direct_link(node.get("inputs", {}).get("model")) == (h3_id, 1)),
        None,
    )
    if second_attention is None:
        return
    second_guider = next(
        (node_id for node_id, node in prompt.items()
         if node.get("class_type") == "BasicGuider"
         and _direct_link(node.get("inputs", {}).get("model")) == (second_attention, 0)),
        None,
    )
    if second_guider is None:
        return
    second_sampler = next(
        (node_id for node_id, node in prompt.items()
         if node.get("class_type") == "SamplerCustomAdvanced"
         and _direct_link(node.get("inputs", {}).get("guider")) == (second_guider, 0)),
        None,
    )
    if second_sampler is None:
        return

    consumers: dict[str, set[str]] = {node_id: set() for node_id in prompt}
    for node_id, node in prompt.items():
        for value in node.get("inputs", {}).values():
            link = _direct_link(value)
            if link and link[0] in consumers:
                consumers[link[0]].add(node_id)

    remove = {second_sampler}
    changed = True
    while changed:
        changed = False
        # Downstream second-pass outputs.
        for parent in tuple(remove):
            for child in consumers.get(parent, ()):
                if child not in remove:
                    remove.add(child)
                    changed = True
        # Upstream nodes used only by the second-pass branch.  Shared first
        # pass nodes (H3 context and first VAE decodes) have another consumer
        # and therefore remain intact.
        for node_id, children in consumers.items():
            if node_id in remove or not children:
                continue
            if children.issubset(remove):
                remove.add(node_id)
                changed = True

    for node_id in remove:
        prompt.pop(node_id, None)


def validate_local_inputs(args: argparse.Namespace) -> None:
    paths = [("--image", args.image), ("--last-image", args.last_image)]
    paths += [("--ref-image", x) for x in args.ref_image]
    paths += [("--ref-video", x) for x in args.ref_video]
    paths += [("--ref-audio", x) for x in args.ref_audio]
    for label, value in paths:
        if value and not Path(value).is_file():
            raise SystemExit(f"{label} 文件不存在：{value}")
    if args.mode == "ref2va":
        if len(args.ref_image) > 9 or len(args.ref_video) > 3 or len(args.ref_audio) > 3:
            raise SystemExit("REF2VA 最多支持 9 张图、3 个视频和 3 个独立音频")
        if not args.ref_image and not args.ref_video:
            raise SystemExit("REF2VA 至少需要一张参考图或一个参考视频（不能只有音频）")


def patch_api_prompt(prompt: dict[str, dict], args: argparse.Namespace, run_id: str) -> dict:
    """Patch the converted API graph by node class, not fragile numeric IDs."""
    # The downloaded UI workflow contains rgthree-only annotation/bypass
    # widgets.  They are not part of the executable graph, and the target
    # deliberately does not install rgthree.  Drop them before comfy run
    # validates the API prompt rather than making the required H3 deployment
    # depend on an unrelated UI extension.
    for node_id, node in list(prompt.items()):
        if node.get("class_type") in {"Label (rgthree)", "Fast Groups Bypasser (rgthree)"}:
            prompt.pop(node_id)

    # v2's refinement branch uses KJNodes' optional NVIDIA VSR resize.  The
    # target has no nvvfx package; the ordinary Lanczos path is the portable
    # fallback and keeps the supplied dual-sampling graph executable.
    for node in prompt.values():
        if node.get("class_type") == "ImageResizeKJv2":
            inputs = node.setdefault("inputs", {})
            if inputs.get("upscale_method") == "nvidia_rtx_vsr":
                inputs["upscale_method"] = "lanczos"

    loaders = [n for n in prompt.values() if n.get("class_type") in {
        "FeiHouEasyH3Loader", "FeiHouEasyH3RemixLoader",
    }]
    if len(loaders) != 1:
        raise RuntimeError(f"工作流需要恰好一个 FeiHou H3 loader，实际 {len(loaders)} 个")
    loader = loaders[0]["inputs"]
    if "remix_model" in loader:
        loader["remix_model"] = REMIX_MODEL
        loader["text_encoder"] = TEXT_ENCODER
        loader["video_vae"] = VIDEO_VAE
        loader["audio_vae"] = AUDIO_VAE
        if "second_sampling_model" in loader:
            loader["second_sampling_model"] = REMIX_MODEL if args.second_sampling else "none"
    else:
        loader["fl2va_model"] = REMIX_MODEL
        loader["ref2va_model"] = REMIX_MODEL
        loader["text_encoder"] = TEXT_ENCODER
        loader["video_vae"] = VIDEO_VAE
        loader["audio_vae"] = AUDIO_VAE
        # The v2 workflow has optional second-pass selectors.  Use the same
        # Remix checkpoint for both roles; it is the only transformer supplied
        # by the requested model page and supports both paths.
        if "custom_second_sampling_models" in loader:
            loader["custom_second_sampling_models"] = bool(args.second_sampling)
        if "second_fl2va_model" in loader:
            loader["second_fl2va_model"] = REMIX_MODEL if args.second_sampling else "none"
        if "second_ref2va_model" in loader:
            loader["second_ref2va_model"] = REMIX_MODEL if args.second_sampling else "none"
        if "second_sampling_use_lora" in loader:
            # The supplied LoRA is the first-pass 8-step Turbo LoRA.  The
            # author's v2 workflow also leaves it off for the refinement pass.
            loader["second_sampling_use_lora"] = False

    lora_nodes = [n for n in prompt.values() if n.get("class_type") == "FeiHouEasyH3LoraStack"]
    if len(lora_nodes) != 1:
        raise RuntimeError(f"工作流需要恰好一个 FeiHou LoRA Stack，实际 {len(lora_nodes)} 个")
    lora_inputs = lora_nodes[0]["inputs"]
    turbo_keys = [
        key for key, value in lora_inputs.items()
        if key.startswith("lora_") and isinstance(value, dict)
        and ("turbo" in str(value.get("lora", "")).lower()
             or "larry" in str(value.get("lora", "")).lower())
    ]
    key = turbo_keys[0] if turbo_keys else "lora_1"
    lora_inputs[key] = {"on": True, "lora": TURBO_LORA, "strength": args.lora_strength}
    if getattr(args, "extra_lora", None):
        # FeiHouEasyH3LoraStack.stack() 收集任意 lora_N 动态槽并合并
        extra_key = "lora_3" if key == "lora_2" else "lora_2"
        lora_inputs[extra_key] = {"on": True, "lora": args.extra_lora,
                                  "strength": args.extra_lora_strength}

    h3_nodes = [n for n in prompt.values() if n.get("class_type") == "FeiHouEasyH3"]
    if len(h3_nodes) != 1:
        raise RuntimeError(f"工作流需要恰好一个 FeiHouEasyH3 主节点，实际 {len(h3_nodes)} 个")
    h3 = h3_nodes[0]["inputs"]
    h3.update({
        "mode": "image" if args.mode == "fl2va" else "reference",
        "prompt": args.prompt,
        "resolution": "custom" if args.size else args.resolution,
        "aspect_ratio": args.aspect,
        "width": args.size[0] if args.size else 1344,
        "height": args.size[1] if args.size else 768,
        # The exported UI workflow has a stale widget value (the old field
        # held the height).  In the current node this is a BOOLEAN; use the
        # explicit duration requested above rather than auto-sizing audio.
        "audio_duration_auto": False,
        "seconds": args.seconds,
        "advanced": True,                 # enables force-offload below
        "fps": 24,
        "keyframe_role": args.keyframe_role,
        "ref_image_size": args.ref_image_size,
        "reference_mention_mode": "index",
        "prompt_optimizer_enabled": bool(args.auto_prompt),
        # The exported workflow came from a ComfyUI instance with a stale
        # provider/scene-guide widget layout.  This target has no prompt API
        # provider configured, so use the node's current valid no-provider
        # value and disable the optional scene guide by default.
        "prompt_optimizer_provider": "",
        "prompt_optimizer_scene_guide": "none",
        "force_offload": True,
        # Keep the H3 block streaming path enabled for the long-sequence,
        # low-VRAM workload.  The target's installed FeiHou node disables
        # only its incompatible legacy final-layer optimization; the stock
        # current-ComfyUI output head remains in use.
        "low_vram_streamed_attention": True,
        "prompt_optimizer_applied": False,
        "second_sampling_output_connected": bool(args.second_sampling),
    })
    # Clear author example media and then fill only the requested media slots.
    for index in range(1, 16):
        h3[f"media_{index}"] = ""
        h3[f"media_type_{index}"] = ""
        h3[f"media_trim_{index}"] = ""
    media = list(args.media_names)
    for index, (name, media_type, trim) in enumerate(media, 1):
        h3[f"media_{index}"] = name
        h3[f"media_type_{index}"] = media_type
        h3[f"media_trim_{index}"] = trim

    # Respect the step split from the supplied full-feature workflow (v2.0 is
    # 12-step first pass + 4-step refinement).  The LoRA is still forced onto
    # the first pass below.  This avoids silently changing the author's
    # quality/speed trade-off.
    for node in prompt.values():
        if node.get("class_type") == "RandomNoise":
            node["inputs"]["noise_seed"] = args.seed
        elif node.get("class_type") == "RandomSeedNoise":
            node["inputs"]["seed"] = args.seed
        elif node.get("class_type") in {"VHS_VideoCombine", "VideoCombineV2"}:
            node["inputs"]["filename_prefix"] = f"H3_{run_id}"
            node["inputs"]["format"] = "video/h264-mp4"
            node["inputs"]["save_output"] = True
            # ComfyUI 0.34 exposes this VHS widget as a BOOLEAN.  Older
            # exported workflows encode the same unchecked/checked value as
            # 0/1, which the current API validator rejects.
            if "audio_duration_auto" in node["inputs"]:
                node["inputs"]["audio_duration_auto"] = bool(
                    node["inputs"]["audio_duration_auto"]
                )

    if not args.second_sampling:
        prune_second_sampling(prompt)

    return prompt


def extract_prompt(preview: Path) -> dict:
    for line in reversed(preview.read_text(errors="replace").splitlines()):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "envelope" and event.get("ok"):
            return event["data"]["prompt"]
    raise RuntimeError(f"无法从 comfy 转换结果取得 API workflow：{preview}")


def ensure_comfy() -> None:
    try:
        ssh("curl -fsS --max-time 5 http://127.0.0.1:8188/system_stats >/dev/null", capture=True)
        return
    except subprocess.CalledProcessError:
        pass
    # VAE stays on GPU: the FeiHou node releases CLIP/VAE from VRAM before
    # first-pass sampling and force_offload unloads the DiT afterwards, so
    # GPU VAE decode is safe on the 32 GB card and ~14x faster than
    # --cpu-vae for long runs (10s/243 frames @480x704: ~53 min -> ~4 min
    # per segment).  The first job after a cold boot may OOM once while
    # ComfyUI's memory manager adapts its DiT offload; retrying the same
    # job succeeds.
    command = (
        f"cd {REMOTE_COMFY} && export PATH={REMOTE_VENV}/bin:$HOME/.local/bin:$PATH; "
        "export XDG_CACHE_HOME=$HOME/autodl-tmp/h3-comfy/cache; "
        "setsid nohup python main.py --listen 127.0.0.1 --port 8188 "
        "--disable-auto-launch --reserve-vram 4 "
        ">/tmp/comfy_boot.log 2>&1 </dev/null & disown"
    )
    ssh(command)
    for _ in range(90):
        time.sleep(2)
        try:
            ssh("curl -fsS --max-time 5 http://127.0.0.1:8188/system_stats >/dev/null", capture=True)
            return
        except subprocess.CalledProcessError:
            continue
    raise RuntimeError("ComfyUI 启动超时；请检查目标机 /tmp/comfy_boot.log")


def check_models(extra_lora: str | None = None) -> None:
    paths = [
        f"$HOME/comfy/ComfyUI/models/diffusion_models/{REMIX_MODEL}",
        f"$HOME/comfy/ComfyUI/models/text_encoders/{TEXT_ENCODER}",
        f"$HOME/comfy/ComfyUI/models/vae/{VIDEO_VAE}",
        f"$HOME/comfy/ComfyUI/models/vae/{AUDIO_VAE}",
        f"$HOME/comfy/ComfyUI/models/loras/{TURBO_LORA}",
    ]
    if extra_lora:
        paths.append(f"$HOME/comfy/ComfyUI/models/loras/{extra_lora}")
    command = "missing=0; " + "; ".join(
        f"test -s {path} || {{ echo MISSING:{path}; missing=1; }}"
        for path in paths
    ) + "; exit $missing"
    try:
        ssh(command)
    except subprocess.CalledProcessError as exc:
        raise SystemExit("目标机模型不齐；请先按 h3-feihou/runbook-h3.md 放置五个文件") from exc


def upload_file(local: str | None, run_id: str, tag: str) -> str | None:
    if not local:
        return None
    name = f"h3_{run_id}_{tag}{Path(local).suffix.lower() or '.bin'}"
    scp_to(Path(local), f"{REMOTE_INPUT}/{name}")
    return name


def collect_media(args: argparse.Namespace, run_id: str) -> list[tuple[str, str, str]]:
    """Upload inputs and return embedded FeiHou media slots."""
    media: list[tuple[str, str, str]] = []
    if args.mode == "fl2va":
        first = upload_file(args.image, run_id, "first")
        last = upload_file(args.last_image, run_id, "last")
        if first:
            media.append((first, "image", ""))
        if last:
            media.append((last, "image", ""))
        return media
    for index, path in enumerate(args.ref_image, 1):
        name = upload_file(path, run_id, f"ref_image_{index}")
        media.append((name, "image", ""))
    for index, path in enumerate(args.ref_video, 1):
        name = upload_file(path, run_id, f"ref_video_{index}")
        media.append((name, "video", ""))
    for index, path in enumerate(args.ref_audio, 1):
        name = upload_file(path, run_id, f"ref_audio_{index}")
        media.append((name, "audio", ""))
    return media


def run_remote_workflow(api_local: Path, run_id: str) -> tuple[str, str]:
    api_remote = f"~/h3-feihou-api-{run_id}.json"
    log_remote = f"/tmp/h3-feihou-{run_id}.log"
    rc_remote = f"$HOME/.h3-feihou-{run_id}.rc"
    scp_to(api_local, api_remote)
    inner = (
        f"export PATH={REMOTE_VENV}/bin:$HOME/.local/bin:$PATH; "
        "export XDG_CACHE_HOME=$HOME/autodl-tmp/h3-comfy/cache; "
        f"comfy run --workflow {api_remote} --host 127.0.0.1 --port 8188 "
        f"--wait --verbose --timeout 10800 > {log_remote} 2>&1; "
        f"echo $? > {rc_remote}"
    )
    ssh(
        f"rm -f {rc_remote}; setsid nohup bash -c {shlex.quote(inner)} "
        "</dev/null >/dev/null 2>&1 & disown; echo submitted"
    )
    print(f"已提交 run_id={run_id}，等待 H3 推理完成…", flush=True)
    for _ in range(1080):
        time.sleep(10)
        try:
            status = ssh(
                f"if test -s {rc_remote}; then printf 'DONE '; cat {rc_remote}; else printf 'RUNNING -'; fi",
                capture=True,
            ).stdout.strip()
        except subprocess.CalledProcessError:
            # A busy target can briefly refuse or drop an SSH session while
            # the ComfyUI worker continues running.  Do not orphan a healthy
            # remote job just because one status probe failed.
            print("  …（SSH 状态探测失败，重试）", flush=True)
            continue
        if status.startswith("DONE "):
            rc = status.split()[-1]
            if rc != "0":
                tail = ssh(f"tail -80 {log_remote}", capture=True).stdout
                raise RuntimeError(f"comfy run 失败 rc={rc}\n{tail}")
            break
        print("  …", flush=True)
    else:
        raise TimeoutError("H3 推理超过 3 小时")
    output = ssh(
        f"ls -t {REMOTE_OUTPUT}/H3_{run_id}* 2>/dev/null | head -1", capture=True
    ).stdout.strip()
    if not output:
        tail = ssh(f"tail -80 {log_remote}", capture=True).stdout
        raise RuntimeError(f"推理成功但找不到视频输出\n{tail}")
    return output, log_remote


def verify_video(path: Path) -> None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        print("!! 本机没有 ffprobe，跳过本地媒体门禁", file=sys.stderr)
        return
    result = subprocess.run(
        [ffprobe, "-v", "error", "-count_frames", "-show_entries",
         "stream=codec_type,codec_name,width,height,nb_read_frames",
         "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1", str(path)],
        text=True, capture_output=True,
    )
    if result.returncode or "codec_type=video" not in result.stdout:
        raise RuntimeError(f"输出视频校验失败：{path}\n{result.stderr}")
    if "codec_type=audio" not in result.stdout:
        print("!! 警告：输出没有音轨；请检查 ComfyUI 日志", file=sys.stderr)
    print(result.stdout.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="MiniMax H3 FeiHou FL2VA/REF2VA runner")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", help="首帧图片")
    parser.add_argument("--last-image", help="末帧图片；可单独提供")
    parser.add_argument("--ref-image", action="append", default=[],
                        help="REF2VA 参考图片，可重复，最多 9 张")
    parser.add_argument("--ref-video", action="append", default=[],
                        help="REF2VA 参考视频，可重复，最多 3 个")
    parser.add_argument("--ref-audio", action="append", default=[],
                        help="REF2VA 独立参考音频，可重复，最多 3 个")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--size", type=parse_size, help="自定义尺寸，例如 1344x768")
    parser.add_argument("--resolution", choices=["360P", "416P", "480P", "540P", "640P", "720P", "768P"], default="480P")
    parser.add_argument("--aspect", choices=["1:1", "3:4", "4:3", "9:16", "16:9", "21:9"], default="16:9")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lora-strength", type=float, default=0.75)
    parser.add_argument("--extra-lora", default=None,
                        help="附加 LoRA（相对 models/loras 的路径，如 H3/xxx.safetensors）")
    parser.add_argument("--extra-lora-strength", type=float, default=1.0)
    parser.add_argument("--second-sampling", action="store_true",
                        help="启用 v2 工作流的二次采样；默认关闭以只使用 Remix 主模型")
    parser.add_argument("--ref-image-size", choices=["match", "480", "544", "640", "736", "768", "832", "928", "1024", "1088"], default="match",
                        help="REF2VA 参考图缩放；match 跟随输出面积")
    parser.add_argument("--auto-prompt", action="store_true", help="启用已在 ComfyUI 设置中配置的提示词 API")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    infer_mode(args)
    validate_local_inputs(args)
    if not 0.2 <= args.seconds <= 30:
        parser.error("--seconds 必须在 0.2 到 30 之间")
    args.seed = args.seed if args.seed is not None else random.randint(0, 2**53 - 1)

    run_id = uuid.uuid4().hex[:8]
    args.remote_image_name = ""
    args.remote_last_image_name = ""
    args.media_names = []
    args.keyframe_role = "last" if args.last_image and not args.image else "first"

    if args.dry_run:
        workflow = json.loads(WORKFLOW.read_text())
        types = [n.get("type") for n in workflow.get("nodes", [])]
        assert "FeiHouEasyH3" in types and "FeiHouEasyH3LoraStack" in types
        assert "FeiHouEasyH3Loader" in types or "FeiHouEasyH3RemixLoader" in types
        second = " + v2 second sampling" if args.second_sampling else ""
        print(f"dry-run OK: {args.mode.upper()} / INT8 ConvRot Remix + INT8 encoder + Turbo LoRA{second}")
        print(f"seed={args.seed} frames={snap_frames(args.seconds)}@24fps run_id={run_id}")
        print(f"workflow={WORKFLOW}")
        return 0

    ensure_comfy()
    check_models(args.extra_lora)
    args.media_names = collect_media(args, run_id)

    ui_remote = f"~/h3-feihou-ui-{run_id}.json"
    ui_local = Path(f"/tmp/h3-feihou-ui-{run_id}.json")
    preview_remote = f"~/h3-feihou-preview-{run_id}.ndjson"
    preview_local = Path(f"/tmp/h3-feihou-preview-{run_id}.ndjson")
    api_local = Path(f"/tmp/h3-feihou-api-{run_id}.json")
    ui_local.write_text(WORKFLOW.read_text())
    scp_to(ui_local, ui_remote)
    ensure = (
        f"export PATH={REMOTE_VENV}/bin:$HOME/.local/bin:$PATH; "
        "export XDG_CACHE_HOME=$HOME/autodl-tmp/h3-comfy/cache; "
        f"comfy run --workflow {ui_remote} --host 127.0.0.1 --port 8188 "
        f"--print-prompt --json > {preview_remote}"
    )
    ssh(ensure)
    # Pull the NDJSON conversion result without making the large graph part of
    # the SSH command line.
    local_run(["scp", *SSH_OPTS, "-P", str(SSH_PORT), f"{SERVER}:{preview_remote}", str(preview_local)])
    prompt = patch_api_prompt(extract_prompt(preview_local), args, run_id)
    api_local.write_text(json.dumps(prompt, ensure_ascii=False, indent=2))
    output_remote, log_remote = run_remote_workflow(api_local, run_id)

    if args.no_download:
        ssh(f"rm -f {REMOTE_INPUT}/h3_{run_id}_*")
        print(f"DONE（未下载）: {output_remote}\n日志: {log_remote}")
        return 0
    out = args.out or Path.home() / "Downloads" / f"h3_{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    local_run(["scp", *SSH_OPTS, "-P", str(SSH_PORT), f"{SERVER}:{output_remote}", str(out)])
    verify_video(out)
    ssh(f"rm -f {REMOTE_INPUT}/h3_{run_id}_*")
    print(f"DONE -> {out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, TimeoutError, subprocess.CalledProcessError) as exc:
        print(f"!! {exc}", file=sys.stderr)
        raise SystemExit(2)
