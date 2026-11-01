#!/usr/bin/env python3
# 用法：在本机运行，上传 3 张图片、1 个参考视频和提示词，下载一采、二采两份有声视频。
# cd /Users/yingtiankai/Desktop/Runbooks/video-generation/h3-replace
'''
uv run --no-project python run.py \
  --images /path/1.png /path/2.png /path/3.png \
  --video /path/reference.mp4 \
  --prompt 'Use <Picture 1>, <Picture 2>, <Picture 3> as references. Follow the motion of <Video 1>.' \
  --seconds 10 --seed 42 --output-dir ./h3-result
'''
# 三张图片需 480p/720p 高；参考视频需 2–15 秒；含空格的路径需加引号。
# --seconds 是生成时长，默认 10 秒（对齐为 243 帧/24fps）；seed 默认 42。
# 输出目录包含 pass1.mp4（576×320）、pass2.mp4（864×480）、result.json 和 run.log，拒绝覆盖目录。
# 需要同目录 run-h3.py、系统 ssh/scp、SSH 密钥、服务器六个 H3 模型和已运行的 ComfyUI。
# 默认服务器：ssh -p 14499 root@connect.weste.seetacloud.com；迁移后修改下方部署常量。
# H3 与 Fibo2 共用服务锁：运行 H3 前先停 Fibo2，再 bash /root/h3-replace/start-comfy.sh。
# 失败/断线不自动重试；服务器任务可能仍在执行，先检查打印的远程 run.log，避免重复提交。
# Upload local H3 references over SSH and download both sampling outputs.

from __future__ import annotations

import argparse
import hashlib
import json
import re
import runpy
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

# shortcut: fixed to this deployment/workflow; update these paths after migration.
HOST = "root@connect.weste.seetacloud.com"
PORT = 14499
REMOTE = "/root/h3-replace"
PYTHON = "/root/h3-venv/bin/python"
COMFY_OUTPUT = "/root/comfy/ComfyUI/output"
SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
               "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3"]
SSH = ["ssh", *SSH_OPTIONS, "-p", str(PORT), HOST]
SCP = ["scp", "-q", *SSH_OPTIONS, "-P", str(PORT)]


def download_result(result: dict, output_dir: Path) -> dict:
    run_id = result.get("run_id")
    outputs = result.get("outputs")
    if (not isinstance(run_id, str) or not re.fullmatch(r"h3_\d{8}_\d{6}_[0-9a-f]{8}", run_id)
            or not isinstance(outputs, list) or len(outputs) != 2
            or [v.get("stage") if isinstance(v, dict) else None for v in outputs] != ["pass1", "pass2"]):
        raise ValueError("Invalid H3 result: expected one run and both sampling outputs")
    paths = []
    for item in outputs:
        path = item.get("path")
        pattern = re.escape(f"{COMFY_OUTPUT}/{run_id}/{item['stage']}") + r"_\d+-audio\.mp4"
        if not isinstance(path, str) or not re.fullmatch(pattern, path):
            raise ValueError("Invalid H3 output path")
        paths.append(path)
    check = subprocess.run([*SSH, shlex.join(["sha256sum", "--", *paths])],
                           check=True, capture_output=True, text=True)
    hashes = {}
    for line in check.stdout.splitlines():
        digest, path = line.split("  ", 1)
        hashes[path] = digest
    if set(hashes) != set(paths) or any(not re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes.values()):
        raise ValueError("Invalid server SHA256 response")
    with tempfile.TemporaryDirectory(prefix=".h3-download-", dir=output_dir) as name:
        stage = Path(name).resolve()
        subprocess.run([*SCP, *(f"{HOST}:{p}" for p in paths), str(stage)], check=True)
        for path in paths:
            digest = hashlib.sha256()
            with (stage / Path(path).name).open("rb") as f:
                for chunk in iter(lambda: f.read(1024**2), b""):
                    digest.update(chunk)
            if digest.hexdigest() != hashes[path]:
                raise ValueError("Downloaded video SHA256 mismatch; server outputs were kept")
        for item, path in zip(outputs, paths):
            target = output_dir / f"{item['stage']}.mp4"
            with (stage / Path(path).name).open("rb") as source, target.open("xb") as dest:
                shutil.copyfileobj(source, dest)
            item.update(server_path=path, path=str(target.resolve()), sha256=hashes[path])
    result.update(server=f"ssh -p {PORT} {HOST}", server_job=f"{REMOTE}/jobs/{run_id}")
    with (output_dir / "result.json").open("x", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def run(args) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite directory: {args.output_dir}")
    if len(args.images) != 3 or not args.prompt.strip():
        raise ValueError("Exactly 3 images, a reference video and a nonempty prompt are required")
    frame_count = runpy.run_path(str(Path(__file__).with_name("run-h3.py")))["frame_count"]
    frames = frame_count(args.seconds)
    if not 0 <= args.seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    sources = [*args.images, args.video]
    for path in sources:
        if not path.is_file() or path.stat().st_size == 0 or not path.suffix:
            raise ValueError(f"Expected a nonempty media file with an extension: {path}")
    preflight = f"""import runpy
ns = runpy.run_path({REMOTE + '/run-h3.py'!r})
missing = [str(ns['COMFY'] / 'models' / ('vae' if role.endswith('_vae') else role) / name)
           for role, name in ns['MODELS'].items()
           if not (ns['COMFY'] / 'models' / ('vae' if role.endswith('_vae') else role) / name).is_file()]
if missing:
    raise SystemExit('H3 not ready; missing models (restore before running):\\n' + '\\n'.join(missing))
ns['request']('/system_stats')
"""
    print("Checking H3 models and ComfyUI (no automatic downloads/start)...", file=sys.stderr, flush=True)
    subprocess.run([*SSH, shlex.join([PYTHON, "-"])], input=preflight,
                   check=True, capture_output=True, text=True)
    args.output_dir.mkdir(parents=True)
    remote_dir = f"{REMOTE}/uploads/local_{uuid.uuid4().hex}"
    print(f"SSH transfer/log directory: {HOST}:{remote_dir}", file=sys.stderr, flush=True)
    subprocess.run([*SSH, f"umask 077; mkdir -p {REMOTE}/uploads && mkdir {remote_dir}"], check=True)
    with tempfile.TemporaryDirectory(prefix=".h3-upload-", dir=args.output_dir) as name:
        stage = Path(name).resolve()
        files = []
        for i, source in enumerate(sources, 1):
            target = stage / f"media_{i}{source.suffix.lower()}"
            shutil.copyfile(source, target)
            files.append(target)
        print("Uploading 3 images and 1 video...", file=sys.stderr, flush=True)
        subprocess.run([*SCP, *map(str, files), f"{HOST}:{remote_dir}/"], check=True)
    remote_files = [f"{remote_dir}/{p.name}" for p in files]
    command = [PYTHON, "-u", f"{REMOTE}/run-h3.py", "--images", *remote_files[:3],
               "--video", remote_files[3], "--prompt", args.prompt, "--seconds", str(args.seconds),
               "--seed", str(args.seed), "--size", args.size]
    log_path = shlex.quote(f"{remote_dir}/run.log")
    shell = ("export PATH=/root/h3-venv/bin:/root/.local/bin:/usr/local/cuda/bin:$PATH; "
             f"setsid nohup {shlex.join(command)} >{log_path} 2>&1 </dev/null & "
             f'pid=$!; status=0; wait "$pid" || status=$?; cat {log_path}; exit "$status"')
    print("Generating both sampling outputs (no automatic retry)...", file=sys.stderr, flush=True)
    completed = subprocess.run([*SSH, shell], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with (args.output_dir / "run.log").open("x", encoding="utf-8") as f:
        f.write(completed.stdout)
    completed.check_returncode()
    _, marker, body = completed.stdout.partition("\n{")
    if not marker:
        raise ValueError("No final H3 result; inspect run.log before retrying")
    result = json.loads("{" + body)
    if not isinstance(result, dict) or result.get("frames") != frames:
        raise ValueError("H3 result frame count does not match the request")
    print("Downloading and verifying both videos...", file=sys.stderr, flush=True)
    result = download_result(result, args.output_dir)
    try:
        subprocess.run([*SSH, shlex.join(["rm", "-rf", "--", remote_dir])], check=True)
    except (OSError, subprocess.CalledProcessError):
        print(f"Warning: outputs saved, but transfer directory was kept: {remote_dir}", file=sys.stderr)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, nargs=3, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", default="576x320", help="first-pass WxH, multiples of 32")
    parser.add_argument("--output-dir", type=Path, default=Path("h3-result"))
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), ensure_ascii=False, indent=2))
    except subprocess.CalledProcessError as exc:
        if exc.stderr or exc.stdout:
            print(exc.stderr or exc.stdout, file=sys.stderr)
        parser.exit(1, "SSH/H3 failed; no automatic retry or remote cleanup. Check the log before resubmitting.\n")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted; the server may still be generating. Inspect remote run.log before retrying.\n")


if __name__ == "__main__":
    main()
