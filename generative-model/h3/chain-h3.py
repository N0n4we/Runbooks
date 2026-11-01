#!/usr/bin/env python3
"""Chain MINIMAX-H3 segments over one original clip, swapping the woman to a reference character.

Runs on the H3 server (media and outputs stay there):

  /root/h3-venv/bin/python chain-h3.py --task task1 --variant fibo2 --size 480x320 \
      --out /root/h3-replace/chain/task1-fibo2 --seconds 10 --deadline-hours 9

Per segment it cuts the reference clip from the original video, feeds
[previous last frame | character sheet | front view] as the three pictures,
generates both sampling passes, keeps the second pass, and records the
first-frame jump against the previous segment. Resumable: existing segments are skipped.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import urllib.request
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/root/h3-replace")
CHAIN = ROOT / "chain"
PY = "/root/h3-venv/bin/python"
PROMPT = (
    "Create one continuous {seconds:g}-second anime shot. Picture 1 is the first frame of the shot: "
    "keep its camera, framing, pose, clothing and lighting unchanged. Picture 2 is the character "
    "sheet and Picture 3 the front view of the character seen in Picture 1. Draw that character with "
    "the face, hairstyle, hair accessories and eye colour of Picture 2 and Picture 3. Follow the "
    "motion, camera and timing of Video 1 exactly and keep its background, its other person and every "
    "piece of cloth unchanged. Do not change the outfit: keep the clothing and accessories of Picture "
    "1 exactly as in Picture 1 and Video 1; never copy clothes, colours or accessories from Picture 2 "
    "or Picture 3. Keep the body state, pose and the contact between the characters exactly as in "
    "Picture 1, and do not cover, move or add anything. Natural movement, no subtitles, no text "
    "overlays, no voice-over."
)


def log(path: Path, message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def ff(*args: str, timeout: int = 900) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", *args],
                   check=True, timeout=timeout)


def canvas(size: tuple[int, int]) -> tuple[int, int]:
    return tuple(round(1.5 * value / 32) * 32 for value in size)  # type: ignore[return-value]


def fit(path: Path, target: Path, size: tuple[int, int], mode: str) -> None:
    """mode=crop fills the canvas from a scene frame; mode=pad keeps a whole reference image."""
    w, h = size
    scale = ("scale=%d:%d:force_original_aspect_ratio=increase,crop=%d:%d" % (w, h, w, h)
             if mode == "crop" else
             "scale=%d:%d:force_original_aspect_ratio=decrease,pad=%d:%d:(ow-iw)/2:(oh-ih)/2:white"
             % (w, h, w, h))
    ff("-i", str(path), "-vf", scale, "-frames:v", "1", str(target))


def frames(path: Path, out_dir: Path) -> tuple[Path, Path]:
    first, last = out_dir / "seg_first.png", out_dir / "seg_last.png"
    ff("-i", str(path), "-vf", "select=eq(n\\,0)", "-frames:v", "1", str(first))
    ff("-sseof", "-0.2", "-i", str(path), "-update", "1", "-q:v", "2", str(last))
    return first, last


def jump(previous: Path, current: Path) -> float:
    a = np.asarray(Image.open(previous).convert("L").resize((96, 64)), dtype=np.float32)
    b = np.asarray(Image.open(current).convert("L").resize((96, 64)), dtype=np.float32)
    return float(np.abs(a - b).mean())


def service_pids() -> list[int]:
    out = subprocess.run(["pgrep", "-f", "main.py --listen 127.0.0.1 --port 8188"],
                         capture_output=True, text=True).stdout
    return [int(value) for value in out.split()]


def service_up() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=5):
            return True
    except OSError:
        return False


def restart_service(log_path: Path) -> None:
    """ComfyUI has died silently under back-to-back H3 jobs; restart it per segment."""
    for pid in service_pids():
        try:
            os.kill(pid, 15)
        except ProcessLookupError:
            pass
    for _ in range(30):
        if not service_pids():
            break
        time.sleep(2)
    lock = "/root/inference-deployment/service.lock"
    for _ in range(60):
        free = subprocess.run(["flock", "-n", lock, "-c", "true"],
                              capture_output=True).returncode == 0
        if free:
            break
        time.sleep(2)
    else:
        raise RuntimeError("service lock still held by the old process")
    env = dict(os.environ)
    env["PATH"] = "/root/h3-venv/bin:/root/.local/bin:/usr/local/cuda/bin:" + env.get("PATH", "")
    with (ROOT / "log" / "comfy-chain.log").open("a") as logfile:
        subprocess.Popen(["setsid", "bash", "-c",
                          "cd /root/h3-replace && exec bash start-comfy.sh"],
                         stdout=logfile, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         env=env, start_new_session=True)
    for _ in range(90):
        if service_up():
            log(log_path, "service restarted")
            return
        time.sleep(4)
    raise RuntimeError("ComfyUI did not become ready")


def generate(images: list[Path], video: Path, seconds: float, seed: int,
             size: str, log_path: Path, prompt: str) -> dict:
    command = [PY, "-u", str(ROOT / "run-h3.py"),
               "--images", *(str(p) for p in images), "--video", str(video),
               "--prompt", prompt, "--seconds", str(seconds),
               "--seed", str(seed), "--size", size]
    env = dict(os.environ)
    env["PATH"] = "/root/h3-venv/bin:/root/.local/bin:/usr/local/cuda/bin:" + env.get("PATH", "")
    result = subprocess.run(command, capture_output=True, text=True, timeout=4 * 3600, env=env)
    log(log_path, f"run-h3 exit={result.returncode} seed={seed}")
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:] or result.stdout[-2000:])
    return json.loads(result.stdout[result.stdout.index("\n{"):])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["task1", "task2", "task3"])
    parser.add_argument("--variant", required=True,
                        help="label used for the input file names, i.e. taskN-<variant>-first.png")
    parser.add_argument("--size", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=9999)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--max-jump", type=float, default=18.0)
    parser.add_argument("--deadline-hours", type=float, default=99)
    parser.add_argument("--first-dup", action="store_true",
                        help="segment 0 sends the given first frame twice as picture 1 and 2")
    parser.add_argument("--first-prompt-file", default="",
                        help="prompt used for segment 0 only")
    parser.add_argument("--prompt-file", default="", help="prompt template, {seconds} allowed")
    parser.add_argument("--max-seconds", type=float, default=0,
                        help="only generate the first N seconds of the source")
    parser.add_argument("--anchor", action="store_true",
                        help="keep the first-frame swap as a fixed picture for every later segment")
    parser.add_argument("--ref-default", default="", help="library image used as the character reference")
    parser.add_argument("--ref-full", default="", help="second library image (usually the full-body view)")
    parser.add_argument("--first-check", type=float, default=0,
                        help="max grey-level jump between the given first frame and segment 0")
    parser.add_argument("--restart-each", action="store_true",
                        help="restart ComfyUI before every segment")
    args = parser.parse_args()

    width, height = (int(v) for v in args.size.split("x"))
    size = canvas((width, height))
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    log_path, state_path = out / "chain.log", out / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"segments": []}
    done = {item["index"] for item in state["segments"]}

    origin = CHAIN / "input" / f"{args.task}-origin.mp4"
    first_source = CHAIN / "input" / f"{args.task}-{args.variant}-first.png"
    sheet = Path(args.ref_default) if args.ref_default else CHAIN / "input" / "sheet.png"
    front = Path(args.ref_full) if args.ref_full else CHAIN / "input" / "front.png"
    for path in (origin, first_source, sheet, front):
        if not path.is_file():
            sys.exit(f"missing input: {path}")
    duration = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
         str(origin)], capture_output=True, text=True, check=True).stdout)
    if args.max_seconds:
        duration = min(duration, args.max_seconds)
    # ceil covers the trailing partial slice (e.g. 238 s -> 24 x 10 s); a last slice under
    # 2 s cannot be a reference clip, so it is dropped instead.
    whole = math.ceil(duration / args.seconds)
    if duration - (whole - 1) * args.seconds < 2:
        whole -= 1
    total = min(args.count, whole - args.start)
    log(log_path, f"{args.task}/{args.variant} canvas={size} segments={total} origin={duration:.2f}s")

    prompt = (Path(args.prompt_file).read_text().strip() if args.prompt_file else PROMPT)
    first_prompt = (Path(args.first_prompt_file).read_text().strip()
                    if args.first_prompt_file else "")
    prompt = prompt.replace("{seconds:g}", f"{args.seconds:g}").replace("{seconds}", f"{args.seconds:g}")
    first_prompt = first_prompt.replace("{seconds:g}", f"{args.seconds:g}").replace("{seconds}", f"{args.seconds:g}")
    reference = [out / "ref_sheet.png", out / "ref_front.png"]
    fit(sheet, reference[0], size, "pad")
    fit(front, reference[1], size, "pad")
    anchor = out / "anchor.png"
    fit(first_source, anchor, size, "crop")
    start = time.monotonic()
    for index in range(args.start, args.start + total):
        if index in done:
            continue
        elapsed = (time.monotonic() - start) / 3600
        if elapsed > args.deadline_hours:
            log(log_path, "deadline reached; stopping")
            break
        work = out / f"seg{index:02d}"
        work.mkdir(exist_ok=True)
        output = out / f"seg{index:02d}.mp4"
        pictures = [work / "picture1.png", *reference]
        if index == 0:
            fit(first_source, work / "picture1.png", size, "crop")
            if args.first_dup:
                pictures = [work / "picture1.png", work / "picture1.png", reference[1]]
        else:
            previous = out / f"seg{index - 1:02d}"
            fit(previous / "seg_last.png", work / "picture1.png", size, "crop")
            if args.anchor:
                pictures = [work / "picture1.png", anchor, reference[0]]
        ref = work / "ref.mp4"
        # one generated segment is 243 frames (10.125 s) for every 10 s of source; assemble-h3.sh
        # retimes the concatenation, so the reference slice keeps 243 frames as well.
        ff("-ss", str(index * args.seconds), "-t", "10.125", "-i", str(origin),
           "-vf", f"fps=24,scale={size[0]}:{size[1]}:force_original_aspect_ratio=increase,"
                  f"crop={size[0]}:{size[1]},setsar=1",
           "-frames:v", "243", "-an", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
           str(ref))
        if args.restart_each:
            restart_service(log_path)
        attempt, entry = None, None
        for seed in (int(s) for s in args.seeds.split(",")):
            started = time.monotonic()
            result = None
            for _ in range(3):
                try:
                    result = generate(pictures, ref, args.seconds, seed, args.size, log_path,
                                      first_prompt if index == 0 and first_prompt else prompt)
                    break
                except RuntimeError as error:
                    log(log_path, f"seg{index:02d} seed={seed} failed: {str(error)[-160:]}")
                    if not service_up():
                        restart_service(log_path)
                    time.sleep(20)
            if result is None:
                raise RuntimeError(f"segment {index} failed 3 times")
            pass2 = Path(next(o["path"] for o in result["outputs"] if o["stage"] == "pass2"))
            ff("-i", str(pass2), "-c", "copy", str(output))
            first, last = frames(output, work)
            if index == 0:
                delta = jump(anchor, first)
            else:
                delta = jump(out / f"seg{index - 1:02d}" / "seg_last.png", first)
            entry = {"index": index, "seed": seed, "run_id": result["run_id"],
                     "minutes": round((time.monotonic() - started) / 60, 2),
                     "jump": None if delta is None else round(delta, 2),
                     "server_output": str(pass2)}
            log(log_path, f"seg{index:02d} seed={seed} {entry['minutes']}min jump={entry['jump']}")
            limit = args.first_check if index == 0 and args.first_check else args.max_jump
            if delta <= limit:
                break
            attempt = entry
        state["segments"].append(attempt or entry)
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=1))
    log(log_path, "done")


if __name__ == "__main__":
    main()
