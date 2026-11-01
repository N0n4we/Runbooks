#!/usr/bin/env python3
"""Run MINIMAX-H3.json on the server; media and outputs stay on that server."""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


WORKFLOW = Path(__file__).with_name("MINIMAX-H3.json")
COMFY = Path.home() / "comfy/ComfyUI"
API = "http://127.0.0.1:8188"
MODELS = {
    "diffusion_models": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "text_encoders": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "video_vae": "minimax_h3_video_vae_fp16.safetensors",
    "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
    "loras": "minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors",
    "latent_upscale_models": "minimax_h3_latent_upscaler_3d_fp16.safetensors",
}


def frame_count(seconds: float) -> int:
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be finite and positive")
    target = max(5, round(seconds * 24))
    return target + (5 - target % 17) % 17


def request(route: str, data: dict | None = None) -> dict:
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(API + route, body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(exc.read().decode(errors="replace")) from exc


def probe(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_entries",
         "stream=codec_name,codec_type,width,height,r_frame_rate,avg_frame_rate,nb_read_frames,duration,sample_rate,channels:format=duration,size",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True, timeout=120,
    )
    return json.loads(result.stdout)


def canvas_size(width: int, height: int) -> tuple[int, int]:
    """Second pass is the upscaler target: 2.25x area of the first pass, aligned to 32."""
    return tuple(round(1.5 * value / 32) * 32 for value in (width, height))


def prepare(source: dict, images: list[str], video: str, prompt: str,
            seconds: float, seed: int, run_id: str,
            width: int = 576, height: int = 320) -> dict:
    workflow = copy.deepcopy(source)
    # shortcut: IDs target this supplied export, update the mapping before using a different workflow.
    nodes = {node["id"]: node for node in workflow["nodes"]}
    if nodes[108]["type"] != "MiniMaxH3ReferenceToVideo":
        raise ValueError("Expected the supplied MINIMAX-H3.json reference workflow")
    for node_id in (132, 129, 130, 131, 135, 136):
        nodes[node_id]["mode"] = 0
    for node_id, image in zip((97, 101, 132), images, strict=True):
        nodes[node_id]["widgets_values"][0] = image
    # shortcut: references are pre-cropped to the generation canvas by the caller.
    for node_id in (99, 102, 129):
        node = nodes[node_id]
        node["type"] = "ImageScale"
        node["properties"] = {"Node name for S&R": "ImageScale"}
        node["inputs"] = [node["inputs"][0]]
        node["outputs"] = [node["outputs"][0]]
        node["widgets_values"] = ["lanczos", *canvas_size(width, height), "center"]
    nodes[135]["widgets_values"].update(
        video=video, force_rate=24, frame_load_cap=0, skip_first_frames=0,
        select_every_nth=1, custom_width=0, custom_height=0,
    )
    nodes[135]["widgets_values"].pop("force_size", None)
    nodes[135]["widgets_values"].pop("choose video to upload", None)
    nodes[8]["widgets_values"][0] = MODELS["diffusion_models"]
    nodes[4]["widgets_values"][0] = MODELS["text_encoders"]
    nodes[5]["widgets_values"][0] = MODELS["video_vae"]
    nodes[6]["widgets_values"][0] = MODELS["audio_vae"]
    nodes[186]["widgets_values"] = [MODELS["loras"], 0.75]
    nodes[346]["widgets_values"] = [prompt]
    nodes[84]["widgets_values"] = [seconds]
    nodes[105]["widgets_values"] = ["16:9 (Widescreen)", 0.18, 32]
    nodes[108]["widgets_values"][1:4] = [width, height, frame_count(seconds)]
    nodes[243]["widgets_values"] = [seed, "fixed"]
    nodes[412]["widgets_values"] = [8, 4096]
    nodes[413]["widgets_values"] = [4]
    # Use PyTorch SDPA with the low-VRAM/FFN patches, without a separately compiled Sage kernel.
    nodes[114]["mode"] = 4
    nodes[416]["mode"] = 4
    # the upscaler's own 32-pixel alignment decides the second-pass width; 0.4 MP is the
    # validated value for 576x320, other first-pass sizes scale by area.
    megapixels = round(0.4 * width * height / (576 * 320), 6)
    nodes[400]["widgets_values"] = [megapixels]
    nodes[417]["widgets_values"] = [
        MODELS["latent_upscale_models"], "megapixels", megapixels, 32, True, True, "cuda", "fp16",
    ]
    for node_id, stage in ((408, "pass1"), (299, "pass2")):
        nodes[node_id]["widgets_values"].update(
            save_output=True, filename_prefix=f"{run_id}/{stage}", frame_rate=24,
        )
    return workflow


def convert(workflow: Path, prompt: str, frames: int, seed: int,
            width: int = 576, height: int = 320) -> dict:
    result = subprocess.run(
        ["comfy", "run", "--workflow", str(workflow), "--host", "127.0.0.1",
         "--port", "8188", "--print-prompt", "--json"],
        capture_output=True, text=True, timeout=180,
    )
    workflow.with_suffix(".conversion.ndjson").write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"comfy conversion failed: {result.stdout}\n{result.stderr}")
    graph = None
    for line in result.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "envelope" and event.get("ok"):
            graph = event["data"]["prompt"]
    if graph is None:
        raise RuntimeError(f"comfy conversion did not return a prompt: {result.stdout}\n{result.stderr}")
    for node_id, node in list(graph.items()):
        if node["class_type"] in {"PreviewImage", "Fast Groups Bypasser (rgthree)", "Note", "MarkdownNote"}:
            del graph[node_id]
    inputs = graph["108"]["inputs"]
    inputs.update(prompt=prompt, width=width, height=height, length=frames)
    for key in list(inputs):
        if key.startswith(("ref_audios.", "ref_video_audios.")) or key in {
            "ref_images.ref_image_3", "ref_images.ref_image_4", "ref_images.ref_image_5",
        }:
            del inputs[key]
    graph["243"]["inputs"]["noise_seed"] = seed
    assert len([k for k in inputs if k.startswith("ref_images.")]) == 3, inputs
    assert len([k for k in inputs if k.startswith("ref_videos.")]) == 1, inputs
    assert all(graph[str(n)]["inputs"]["save_output"] for n in (408, 299))
    assert all(graph[str(n)]["class_type"] == "SamplerCustomAdvanced" for n in (247, 295))
    return graph


def self_test() -> None:
    source = json.loads(WORKFLOW.read_text())
    original = copy.deepcopy(source)
    workflow = prepare(source, ["one.png", "two.png", "three.png"], "video.mp4",
                       "test <Picture 1> <Picture 2> <Picture 3> <Video 1>", 10, 42, "self-test")
    nodes = {n["id"]: n for n in workflow["nodes"]}
    assert source == original
    assert [frame_count(x) for x in (10, 15, 20)] == [243, 362, 481]
    for value in (0, -1, float("nan"), float("inf")):
        try:
            frame_count(value)
        except ValueError:
            pass
        else:
            raise AssertionError(value)
    assert all(nodes[n]["mode"] == 0 for n in (97, 101, 132, 131, 135, 136, 247, 295))
    assert all(nodes[n]["type"] == "ImageScale" for n in (99, 102, 129))
    assert nodes[412]["widgets_values"][0] > 1 and nodes[413]["widgets_values"][0] > 1
    assert nodes[4]["widgets_values"][0] == MODELS["text_encoders"]
    assert nodes[8]["widgets_values"][0] == MODELS["diffusion_models"]
    assert nodes[417]["widgets_values"][4:6] == [True, True]
    assert nodes[114]["mode"] == 4
    assert nodes[105]["widgets_values"][1:] == [0.18, 32]
    assert nodes[108]["widgets_values"][1:4] == [576, 320, 243]
    assert "force_size" not in nodes[135]["widgets_values"]
    assert all(nodes[n]["widgets_values"]["save_output"] for n in (408, 299))
    print("self-test OK: original unchanged; 3 images + video; two samplers and saved outputs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, nargs=3, metavar=("IMAGE1", "IMAGE2", "IMAGE3"))
    parser.add_argument("--video", type=Path)
    parser.add_argument("--prompt")
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--size", default="576x320", metavar="WxH",
                        help="first-pass size, multiples of 32; the second pass is 1.5x")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and convert without submitting")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.images or not args.video or not args.prompt or not args.prompt.strip():
        parser.error("--images (3 server paths), --video and --prompt are required")
    try:
        width, height = (int(v) for v in args.size.lower().split("x", 1))
    except ValueError:
        parser.error("--size must look like 576x320")
    if width % 32 or height % 32 or width < 64 or height < 64:
        parser.error("--size must use multiples of 32 and be at least 64x64")
    frames = frame_count(args.seconds)
    if not 0 <= args.seed < 2**64:
        parser.error("seed must be an unsigned 64-bit integer")
    input_probes = []
    for index, path in enumerate([*args.images, args.video]):
        if not path.is_file():
            parser.error(f"Server input does not exist: {path}")
        info = probe(path)
        stream = next(s for s in info["streams"] if s["codec_type"] == "video")
        if index < 3 and (stream["width"], stream["height"]) != canvas_size(width, height):
            parser.error(f"Reference image must be the generation canvas "
                         f"{canvas_size(width, height)[0]}x{canvas_size(width, height)[1]}: {path}")
        if index == 3 and not 2 <= float(info["format"]["duration"]) <= 15:
            parser.error("Reference video must be 2-15 seconds (the supplied workflow's limit)")
        input_probes.append({"path": str(path.resolve()), "ffprobe": info})
    for role, name in MODELS.items():
        directory = "vae" if role.endswith("_vae") else role
        if not (COMFY / "models" / directory / name).is_file():
            parser.error(f"Missing model: {directory}/{name}")
    request("/system_stats")
    run_id = "h3_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    job_dir = Path(__file__).resolve().parent / "jobs" / run_id
    job_dir.mkdir(parents=True)
    input_dir = COMFY / "input" / run_id
    input_dir.mkdir(parents=True)
    names = []
    for index, path in enumerate([*args.images, args.video]):
        name = f"media_{index + 1}{path.suffix.lower()}"
        shutil.copy2(path, input_dir / name)
        names.append(f"{run_id}/{name}")
    source = json.loads(WORKFLOW.read_text())
    workflow = prepare(source, names[:3], names[3], args.prompt, args.seconds, args.seed, run_id,
                       width, height)
    ui_path = job_dir / "workflow.ui.json"
    ui_path.write_text(json.dumps(workflow, ensure_ascii=False, indent=2))
    graph = convert(ui_path, args.prompt, frames, args.seed, width, height)
    (job_dir / "workflow.api.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2))
    (job_dir / "inputs.json").write_text(json.dumps(input_probes, indent=2))
    if args.dry_run:
        print(f"dry-run OK: {run_id}, {frames} frames, 3 images + 1 video, both sampling outputs\n{job_dir}")
        return
    response = request("/prompt", {"prompt": graph, "client_id": run_id,
                                   "extra_data": {"extra_pnginfo": {"workflow": workflow}}})
    prompt_id = response["prompt_id"]
    (job_dir / "submission.json").write_text(json.dumps(response, indent=2))
    print(f"submitted prompt_id={prompt_id} run_id={run_id} frames={frames}", flush=True)
    deadline = time.monotonic() + 10800
    while time.monotonic() < deadline:
        history = request(f"/history/{prompt_id}").get(prompt_id)
        if history:
            (job_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2))
            if history["status"]["status_str"] == "error":
                raise RuntimeError(json.dumps(history["status"], ensure_ascii=False))
            if history["status"]["completed"]:
                break
        time.sleep(5)
    else:
        raise TimeoutError(f"3-hour deadline exceeded; inspect prompt {prompt_id}; it was not resubmitted")
    outputs = []
    for node_id, stage, size in (("408", "pass1", (width, height)),
                                 ("299", "pass2", None)):
        videos = history["outputs"][node_id]["gifs"]
        item = next(v for v in videos if v["filename"].endswith("-audio.mp4"))
        if item["type"] != "output":
            raise RuntimeError(f"{stage} was not saved as a persistent output: {item}")
        output = COMFY / "output" / item["subfolder"] / item["filename"]
        info = probe(output)
        video = next(s for s in info["streams"] if s["codec_type"] == "video")
        audio = next(s for s in info["streams"] if s["codec_type"] == "audio")
        got = (int(video["width"]), int(video["height"]))
        # shortcut: the upscaler picks the second-pass width itself (32-pixel alignment), so only
        # the aspect ratio is checked there; the first pass keeps the exact requested size.
        close = size is not None or abs(got[0] - round(got[1] * width / height)) <= 32
        if (int(video["nb_read_frames"]) != frames or (size is not None and got != size) or not close
                or video["avg_frame_rate"] != "24/1" or video["codec_name"] != "h264" or audio["codec_name"] != "aac"):
            raise RuntimeError(f"Output validation failed: {output}\n{info}")
        outputs.append({"stage": stage, "path": str(output), "ffprobe": info})
    result = {"prompt_id": prompt_id, "run_id": run_id, "frames": frames, "outputs": outputs}
    (job_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
