#!/usr/bin/env python3
r"""Qwen-Image-2.1-Turbo：服务器常驻服务 / 标准库 HTTP 客户端。
服务器启动：/root/fibo2-venv/bin/python run-qwenimage.py --serve --port 8191
服务器调用：python run-qwenimage.py --images one.png two.png --prompt 'Combine <image1> and <image2>' --output result.png
文生图：python run-qwenimage.py --prompt '一只坐在窗边的橘猫，柔和自然光，摄影作品' --output cat.png
完全本地推理，固定官方 8 步 / CFG=1；本机 SSH 上传下载请使用同目录 run.py。
"""

import argparse
import base64
import binascii
import hashlib
import io
import json
import os
import re
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from functools import partial
from pathlib import Path

MODEL_ID = "Qwen/Qwen-Image-2.1-Turbo"
MODEL_REVISION = "4954d69c119759f085aa36bd5e66e27647cdee60"
MODEL_DIR = Path("/root/autodl-tmp/qwenimage2.1/models/Qwen-Image-2.1-Turbo")
SIGMAS = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]
MAX_BODY = 64 * 1024**2


def chunked_ffn(forward, hidden_states):
    import torch
    # shortcut: token-wise QwenImage21 SwiGLU only; revalidate when changing the pinned architecture.
    return torch.cat([forward(chunk) for chunk in hidden_states.split(1024, dim=1)], dim=1)


def validate(payload: dict) -> tuple:
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    fields = {"prompt", "width", "height", "seed", *(f"image_{i}" for i in range(1, 5))}
    if payload.keys() - fields:
        raise ValueError("Unknown request fields: " + ", ".join(sorted(payload.keys() - fields)))
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt.encode()) > 16384:
        raise ValueError("prompt must be nonempty UTF-8 text, at most 16384 bytes")
    names = [f"image_{i}" for i in range(1, 5) if f"image_{i}" in payload]
    if names != [f"image_{i}" for i in range(1, len(names) + 1)]:
        raise ValueError("Provide 0-4 contiguous image fields starting at image_1")
    if set(re.findall(r"<image_?(\d+)>", prompt)) - {str(i) for i in range(1, len(names) + 1)}:
        raise ValueError("Prompt references an image that was not supplied")
    images = []
    for name in names:
        try:
            if not isinstance(payload[name], str):
                raise ValueError("Image must be base64 text")
            raw = base64.b64decode(payload[name], validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError(f"{name}: invalid base64") from exc
        if not 1 <= len(raw) <= 10 * 1024**2:
            raise ValueError(f"{name}: decoded file must be 1 byte to 10 MiB")
        images.append(raw)
    width, height, seed = (payload.get(k, v) for k, v in (("width", 1024), ("height", 1024), ("seed", 42)))
    if any(type(v) is not int or not 512 <= v <= 2048 or v % 32 for v in (width, height)):
        raise ValueError("width and height must be multiples of 32, from 512 to 2048")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    return prompt, images, width, height, seed


def serve(model_dir: Path, port: int) -> None:
    # All model/config/font reads are local; never use an external captioning API.
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from diffusers import QwenImage21Pipeline
    from accelerate import cpu_offload_with_hook
    from PIL import Image, ImageOps, UnidentifiedImageError

    if not (model_dir / "model_index.json").is_file():
        raise FileNotFoundError(f"Missing local model: {model_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this deployment")
    pipe = QwenImage21Pipeline.from_pretrained(model_dir, dtype=torch.bfloat16, local_files_only=True)
    if list(pipe.config.sample_sigmas or []) != SIGMAS:
        raise RuntimeError("Checkpoint does not contain the expected official 8-step Turbo schedule")
    if any(module.dtype != torch.bfloat16 for module in (pipe.text_encoder, pipe.transformer, pipe.vae)):
        raise RuntimeError("Expected BF16 model components; refusing an unvalidated VRAM profile")
    for block in pipe.transformer.transformer_blocks:
        block.img_mlp.forward = partial(chunked_ffn, block.img_mlp.forward)
    _, text_hook = cpu_offload_with_hook(pipe.text_encoder, torch.device("cuda"))
    _, vae_hook = cpu_offload_with_hook(pipe.vae, torch.device("cuda"), prev_module_hook=text_hook)

    def release_encoders(module, args):
        text_hook.offload()
        vae_hook.offload()

    pipe.transformer.register_forward_pre_hook(release_encoders)
    pipe.transformer.enable_group_offload(onload_device=torch.device("cuda"),
                                        offload_device=torch.device("cpu"),
                                        offload_type="block_level", num_blocks_per_group=4,
                                        use_stream=False)
    pipe.vae.enable_tiling()
    jobs = Path(__file__).resolve().parent / "jobs"
    jobs.mkdir(exist_ok=True)
    lock = threading.Lock()

    def generate(payload: dict) -> tuple:
        prompt, sources, width, height, seed = validate(payload)
        images = []
        for i, raw in enumerate(sources, 1):
            try:
                with Image.open(io.BytesIO(raw)) as image:
                    if image.format not in {"PNG", "JPEG", "WEBP"} or getattr(image, "n_frames", 1) != 1:
                        raise ValueError("Expected a single-frame PNG, JPEG or WebP")
                    if image.width * image.height > 16_000_000:
                        raise ValueError("Image exceeds 16 megapixels")
                    images.append(ImageOps.exif_transpose(image).convert("RGBA"))
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
                raise ValueError(f"image_{i}: invalid or oversized raster image") from exc
        job_id = "qwenimage_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        job = jobs / job_id
        job.mkdir()
        record = dict(job_id=job_id, model=MODEL_ID, prompt=prompt, width=width, height=height,
                      seed=seed, steps=8, true_cfg_scale=1.0, image_count=len(images), inputs=[],
                      model_revision=MODEL_REVISION, sample_sigmas=SIGMAS, use_kv_cache=True,
                      mode="edit" if images else "text-to-image", reference_resolution=1024,
                      ffn_chunk_tokens=1024, offload="transformer: 4-block CPU/GPU groups; encoder/VAE: component",
                      allocator=os.environ["PYTORCH_CUDA_ALLOC_CONF"])
        # Keep the previous client's <image_1> spelling compatible with native Qwen <image1>.
        effective_prompt = re.sub(r"<image_(\d+)>", lambda m: f"<image{m[1]}>", prompt)
        record["effective_prompt"] = effective_prompt
        for i, (raw, image) in enumerate(zip(sources, images), 1):
            (job / f"image_{i}.source").write_bytes(raw)
            image.save(job / f"image_{i}.png")
            record["inputs"].append(dict(label=f"<image{i}>", size=list(image.size),
                                         source_sha256=hashlib.sha256(raw).hexdigest()))
        (job / "request.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        started = time.perf_counter()
        steps = []

        def progress(pipeline, step, timestep, values):
            steps.append(dict(step=step + 1, elapsed_seconds=time.perf_counter() - started))
            print(json.dumps(dict(job_id=job_id, **steps[-1])), flush=True)
            return values

        torch.cuda.reset_peak_memory_stats()
        try:
            with torch.inference_mode():
                image = pipe(prompt=effective_prompt, image=images or None, width=width, height=height,
                             true_cfg_scale=1.0, use_kv_cache=True, output_resolution=1024,
                             generator=torch.Generator(device="cpu").manual_seed(seed),
                             callback_on_step_end=progress).images[0]
            torch.cuda.synchronize()
            assert image.size == (width, height) and len(steps) == 8
            image.save(job / "output.png")
            body = (job / "output.png").read_bytes()
            record.update(elapsed_seconds=time.perf_counter() - started, progress=steps,
                          output_sha256=hashlib.sha256(body).hexdigest(), output_bytes=len(body), output_mode=image.mode,
                          torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          torch_peak_reserved_bytes=torch.cuda.max_memory_reserved())
            (job / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
            return body, record
        except Exception as exc:
            (job / "error.json").write_text(json.dumps(dict(job_id=job_id, error=str(exc)), indent=2) + "\n")
            raise
        finally:
            pipe.maybe_free_model_hooks()
            text_hook.offload()
            vae_hook.offload()
            torch.cuda.empty_cache()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body, content_type="application/json", headers=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for name, value in (headers or {}).items():
                self.send_header(name, str(value))
            self.end_headers()
            self.wfile.write(body)

        def error(self, status, message):
            self.reply(status, json.dumps({"error": message}, ensure_ascii=False).encode())

        def do_GET(self):
            if self.path != "/health":
                self.error(404, "Not found")
                return
            self.reply(200, json.dumps(dict(ready=True, model=MODEL_ID, busy=lock.locked(),
                                           gpu=torch.cuda.get_device_name(0), local_only=True)).encode())

        def do_POST(self):
            if self.path != "/generate":
                self.error(404, "Not found")
                return
            if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                self.error(415, "Expected application/json")
                return
            if not lock.acquire(blocking=False):
                self.error(503, "GPU is busy; this service runs one request at a time")
                return
            try:
                self.connection.settimeout(120)
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_BODY:
                    raise ValueError("Request body must be 1 byte to 64 MiB")
                raw = self.rfile.read(size)
                if len(raw) != size:
                    raise ValueError("Incomplete request body")
                body, result = generate(json.loads(raw))
                self.reply(200, body, "image/png", {"X-QwenImage-Job-ID": result["job_id"],
                           "X-Elapsed-Seconds": f'{result["elapsed_seconds"]:.3f}'})
            except (ValueError, TimeoutError) as exc:
                self.error(400, str(exc))
            except Exception as exc:
                traceback.print_exc()
                self.error(500, str(exc))
            finally:
                lock.release()

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"READY {MODEL_ID} http://127.0.0.1:{port}; local-only; 4-block CPU/GPU offload; official 8 steps; KV cache; FFN chunk=1024", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def client(args) -> None:
    if len(args.images) > 4 or not args.prompt:
        raise ValueError("--prompt is required; --images accepts 1-4 optional references")
    if args.output.suffix.lower() != ".png":
        raise ValueError("--output must end in .png")
    metadata = args.output.with_suffix(".json")
    if args.output.exists() or metadata.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output} or {metadata}")
    for path in args.images:
        if not path.is_file() or not 1 <= path.stat().st_size <= 10 * 1024**2:
            raise ValueError(f"Expected an image file of 1 byte to 10 MiB: {path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(prompt=args.prompt, width=args.width, height=args.height, seed=args.seed)
    payload.update({f"image_{i}": base64.b64encode(path.read_bytes()).decode()
                    for i, path in enumerate(args.images, 1)})
    validate(payload)
    raw = json.dumps(payload, ensure_ascii=False).encode()
    if len(raw) > MAX_BODY:
        raise ValueError("Request exceeds 64 MiB")
    request = urllib.request.Request(args.url.rstrip("/") + "/generate", raw,
                                     {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            body = response.read()
            if response.headers.get_content_type() != "image/png" or not body.startswith(b"\x89PNG\r\n\x1a\n"):
                raise RuntimeError("Service did not return a PNG image")
            result = dict(job_id=response.headers.get("X-QwenImage-Job-ID"),
                          elapsed_seconds=response.headers.get("X-Elapsed-Seconds"),
                          output=str(args.output.resolve()), sha256=hashlib.sha256(body).hexdigest())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read(8192).decode(errors='replace')}") from exc
    with args.output.open("xb") as f:
        f.write(body)
    with metadata.open("x") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def self_test() -> None:
    sample = base64.b64encode(b"payload-shape-test-only").decode()
    assert len(validate(dict(prompt="一只坐在窗边的橘猫"))[1]) == 0
    one = dict(prompt="Edit <image1>", image_1=sample)
    assert len(validate(one)[1]) == 1
    four = dict(prompt="Combine <image_1> through <image_4>",
                **{f"image_{i}": sample for i in range(1, 5)})
    assert len(validate(four)[1]) == 4
    for count in range(1, 5):
        assert len(validate(dict(prompt=f"Use <image{count}>",
                                 **{f"image_{i}": sample for i in range(1, count + 1)}))[1]) == count
    assert validate(dict(prompt="poster", width=2048, height=2048))[2:4] == (2048, 2048)
    for bad in ({}, {**one, "image_3": sample}, {**one, "image_5": sample},
                {**one, "prompt": "Use <image_2>"}, {**one, "width": 769},
                {**one, "seed": -1}, {**one, "seed": True}, {**one, "image_1": "!invalid"}):
        try:
            validate(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)
    print("self-test OK: text-to-image, 1-4 ordered images, native/compatible tags, bounds and invalid payloads")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--port", type=int, default=8191)
    parser.add_argument("--url", default="http://127.0.0.1:8191")
    parser.add_argument("--images", type=Path, nargs="+", default=[])
    parser.add_argument("--prompt")
    parser.add_argument("--output", type=Path, default=Path("qwenimage.png"))
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.serve:
        serve(args.model_dir, args.port)
    else:
        try:
            client(args)
        except (ValueError, OSError, RuntimeError) as exc:
            parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
