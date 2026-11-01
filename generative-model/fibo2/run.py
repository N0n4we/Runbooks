#!/usr/bin/env python3
# 用法：在本机运行，自动 SCP 上传本地图片、SSH 生成、SCP 下载结果，无需手动建隧道。
# cd /Users/yingtiankai/Desktop/Runbooks/video-generation/fibo2
'''
uv run --no-project python run.py \
  --images /path/1.png /path/2.png /path/3.png /path/4.png \
  --prompt '将 <image_1>、<image_2>、<image_3>、<image_4> 按顺序排成四格分镜' \
  --width 1024 --height 1024 --seed 42 --output result.png
'''
# --images 支持 1–4 张，标签按文件顺序编号；单图仅传一个路径，并只引用 <image_1>。
# 含空格的路径需加引号。
# 默认 768×768、seed=42；输出 PNG 和同名 JSON，已有输出时拒绝覆盖。
# 需要同目录 run-fibo2.py、系统 ssh/scp、已配置的 SSH 密钥，以及服务器已运行的 Fibo2 服务。
# 默认服务器：ssh -p 14499 root@connect.weste.seetacloud.com（2026-10-31 迁移，旧机已删除）；迁移后修改下方部署常量。
# 断线/失败不自动重试；先检查打印的远程目录，避免重复生成。
# Upload local images over SSH, call the deployed Fibo2 service, and download the result.

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import runpy
import shlex
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

# shortcut: fixed to this deployment; update these constants after a server migration.
HOST = "root@connect.weste.seetacloud.com"
PORT = 14499
REMOTE = "/root/fibo2"
SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
               "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3"]
SSH = ["ssh", *SSH_OPTIONS, "-p", str(PORT), HOST]
SCP = ["scp", "-q", *SSH_OPTIONS, "-P", str(PORT)]


def run(args) -> dict:
    metadata = args.output.with_suffix(".json")
    if args.output.suffix.lower() != ".png":
        raise ValueError("--output must end in .png")
    if args.output.exists() or metadata.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output} or {metadata}")
    if not 1 <= len(args.images) <= 4:
        raise ValueError("--images requires 1-4 files")
    payload = dict(prompt=args.prompt, width=args.width, height=args.height, seed=args.seed)
    for i, path in enumerate(args.images, 1):
        if not path.is_file():
            raise ValueError(f"Not an image file: {path}")
        with path.open("rb") as f:
            payload[f"image_{i}"] = base64.b64encode(f.read(10 * 1024**2 + 1)).decode()
    validate = runpy.run_path(str(Path(__file__).with_name("run-fibo2.py")))["validate"]
    prompt, sources, width, height, seed = validate(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    remote_dir = f"{REMOTE}/uploads/local_{uuid.uuid4().hex}"
    print(f"SSH transfer directory: {HOST}:{remote_dir}", file=sys.stderr, flush=True)
    subprocess.run([*SSH, f"umask 077; mkdir -p {REMOTE}/uploads && mkdir {remote_dir}"], check=True)
    with tempfile.TemporaryDirectory(prefix=".fibo2-transfer-", dir=args.output.parent) as name:
        stage = Path(name).resolve()
        files = []
        for i, source in enumerate(sources, 1):
            path = stage / f"image_{i}.input"
            path.write_bytes(source)
            files.append(path)
        print(f"Uploading {len(files)} image(s)...", file=sys.stderr, flush=True)
        subprocess.run([*SCP, *map(str, files), f"{HOST}:{remote_dir}/"], check=True)
        command = ["/root/fibo2-venv/bin/python", f"{REMOTE}/run-fibo2.py", "--images",
                   *(f"{remote_dir}/{p.name}" for p in files), "--prompt", prompt,
                   "--width", str(width), "--height", str(height), "--seed", str(seed),
                   "--output", f"{remote_dir}/output.png"]
        print("Generating (no automatic retry)...", file=sys.stderr, flush=True)
        subprocess.run([*SSH, shlex.join(command)], check=True, stdout=subprocess.DEVNULL)
        print("Downloading result...", file=sys.stderr, flush=True)
        subprocess.run([*SCP, f"{HOST}:{remote_dir}/output.png",
                        f"{HOST}:{remote_dir}/output.json", str(stage)], check=True)
        image = (stage / "output.png").read_bytes()
        result = json.loads((stage / "output.json").read_text())
        if (not image.startswith(b"\x89PNG\r\n\x1a\n") or not isinstance(result, dict)
                or not isinstance(result.get("job_id"), str)
                or result.get("sha256") != hashlib.sha256(image).hexdigest()):
            raise ValueError("Downloaded PNG/metadata failed verification; remote files were kept")
        result.update(output=str(args.output.resolve()), server=f"ssh -p {PORT} {HOST}",
                      server_output=f"{REMOTE}/jobs/{result['job_id']}/output.png")
        with args.output.open("xb") as f:
            f.write(image)
        with metadata.open("x", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    try:
        subprocess.run([*SSH, shlex.join(["rm", "-rf", "--", remote_dir])], check=True)
    except (OSError, subprocess.CalledProcessError):
        print(f"Warning: result saved, but transfer directory was kept: {remote_dir}", file=sys.stderr)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, default=Path("result.png"))
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), ensure_ascii=False, indent=2))
    except subprocess.CalledProcessError as exc:
        parser.exit(1, f"SSH/SCP failed (exit {exc.returncode}); remote files were kept. "
                       "Check the transfer directory printed above before retrying inference.\n")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted; the server may still be generating. Check remote files before retrying.\n")


if __name__ == "__main__":
    main()
