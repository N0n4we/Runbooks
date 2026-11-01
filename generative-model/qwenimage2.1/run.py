#!/usr/bin/env python3
# 用法：在本机运行，自动 SCP 上传本地图片、SSH 调用 Qwen 服务、SCP 下载 PNG，无需手动隧道。
# cd /Users/yingtiankai/Desktop/Runbooks/video-generation/qwenimage2.1
'''
uv run --no-project python run.py \
  --images /path/1.png /path/2.png /path/3.png /path/4.png \
  --prompt '将 <image1>、<image2>、<image3>、<image4> 按顺序排成四格分镜' \
  --width 1024 --height 1024 --seed 42 --output result.png
'''
# --images 可传 1–4 张，标签按文件顺序编号；不传 --images 则文生图，单图仅引用 <image1>。
# 含空格的路径需加引号。
# 默认 1024×1024、seed=42；固定官方 8 步 / CFG=1 / KV cache，输出 PNG 和同名 JSON，拒绝覆盖。
# 需要同目录 run-qwenimage.py、系统 ssh/scp、SSH 密钥，以及已运行的 Qwen 服务；不向外部 API 发图。
# 默认服务器：ssh -p 30433 root@connect.nmb2.seetacloud.com；迁移后修改下方部署常量。
# 断线/失败不自动重试；先检查打印的远程目录，避免重复生成。
# 文生图例子：uv run --no-project python run.py --prompt '一只坐在窗边的橘猫，摄影作品' --output cat.png

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
HOST = "root@connect.nmb2.seetacloud.com"
PORT = 30433
REMOTE = "/root/qwenimage2.1"
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
    if len(args.images) > 4:
        raise ValueError("--images accepts at most 4 files")
    payload = dict(prompt=args.prompt, width=args.width, height=args.height, seed=args.seed)
    for i, path in enumerate(args.images, 1):
        if not path.is_file():
            raise ValueError(f"Not an image file: {path}")
        with path.open("rb") as f:
            payload[f"image_{i}"] = base64.b64encode(f.read(10 * 1024**2 + 1)).decode()
    validate = runpy.run_path(str(Path(__file__).with_name("run-qwenimage.py")))["validate"]
    prompt, sources, width, height, seed = validate(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    remote_dir = f"{REMOTE}/uploads/local_{uuid.uuid4().hex}"
    print(f"SSH transfer directory: {HOST}:{remote_dir}", file=sys.stderr, flush=True)
    subprocess.run([*SSH, f"umask 077; mkdir -p {REMOTE}/uploads && mkdir {remote_dir}"], check=True)
    with tempfile.TemporaryDirectory(prefix=".qwenimage-transfer-", dir=args.output.parent) as name:
        stage = Path(name).resolve()
        files = []
        for i, source in enumerate(sources, 1):
            path = stage / f"image_{i}.input"
            path.write_bytes(source)
            files.append(path)
        print(f"Uploading {len(files)} image(s)...", file=sys.stderr, flush=True)
        if files:
            subprocess.run([*SCP, *map(str, files), f"{HOST}:{remote_dir}/"], check=True)
        command = ["/root/fibo2-venv/bin/python", f"{REMOTE}/run-qwenimage.py",
                   *(["--images", *(f"{remote_dir}/{p.name}" for p in files)] if files else []), "--prompt", prompt,
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
    parser.add_argument("--images", type=Path, nargs="+", default=[])
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, default=Path("result.png"))
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
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
