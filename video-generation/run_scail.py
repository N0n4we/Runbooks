#!/usr/bin/env python3
"""SCAIL-2 动作迁移一键调度脚本（本地驱动，远程执行）。

四个主要输入项全部参数化：
  --video    驱动视频（动作来源）
  --image    单图参考（角色外观，画布自动跟随其尺寸）
  --images   多图参考（可选，最多 5 张，自动解除 bypass）
  --prompt   正面提示词（默认沿用工作流里的）

用法示例：
  python3 scail_run.py --video ~/Documents/Games/ref2.mp4 --image ~/Documents/Games/input.png \
      --prompt "一个女人在跳舞" --out ~/Downloads/out.mp4
  python3 scail_run.py --video ref.mp4 --image a.png --images b.png c.png --fps 30 --pose-strength 0.8

依赖：ssh/scp 免密、服务器已按 runbook-scail.md 部署（ComfyUI 在 8188）。
"""
import argparse, json, subprocess, sys, time, urllib.request
from pathlib import Path

# ---------- 服务器常量（wp08:15024 实测；换机器改这里或用参数覆盖） ----------
DEF_SERVER = "gJkaKP@wp08.unicorn.org.cn"
DEF_SSH_PORT = 15024
SVR_INPUT = "~/comfy/ComfyUI/input"
SVR_OUTGLOB = "~/comfy/ComfyUI/output/%date:yyyy-MM-dd%"
SHARED_JSON = Path(__file__).parent / "Scail动作迁移工作流.json"

# 多图参考链路节点（482-486 LoadImage, 481 ImageBatchMulti, 480 ImageResize, 453 SAM3 track）
MULTI_NODES = [482, 483, 484, 485, 486]
MULTI_ENABLE = MULTI_NODES + [481, 480, 453]


def sh(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def ssh(server, port, remote_cmd, capture=False):
    base = ["ssh", "-o", "StrictHostKeyChecking=accept-new", "-p", str(port), server, remote_cmd]
    return subprocess.run(base, check=True, capture_output=capture, text=True)


def scp_to(server, port, local, remote):
    sh(["scp", "-o", "StrictHostKeyChecking=accept-new", "-P", str(port), local, f"{server}:{remote}"])


def main():
    ap = argparse.ArgumentParser(description="SCAIL-2 动作迁移调度")
    ap.add_argument("--video", required=True, help="驱动视频路径")
    ap.add_argument("--image", required=True, help="单图参考路径（角色外观/画布来源）")
    ap.add_argument("--images", nargs="*", default=[], help="多图参考（最多5张，可选）")
    ap.add_argument("--prompt", default=None, help="正面提示词（默认沿用工作流）")
    ap.add_argument("--negprompt", default=None, help="负面提示词（默认沿用工作流）")
    ap.add_argument("--fps", type=int, default=0, help="输出帧率 force_rate，0=跟随源视频")
    ap.add_argument("--cap", type=int, default=0, help="帧数上限 frame_load_cap，0=全帧")
    ap.add_argument("--skip", type=int, default=0, help="跳过开头帧数 skip_first_frames")
    ap.add_argument("--pose-strength", type=float, default=None, help="动作参考强度（默认工作流值1）")
    ap.add_argument("--ref-strength", type=float, default=None, help="外观参考强度（默认工作流值1）")
    ap.add_argument("--cfg", type=float, default=None, help="条件引导强度（默认工作流值）")
    ap.add_argument("--object-indices", default=None,
                    help="迁移对象编号，逗号分隔如 '0,2'；空=全部。编号顺序由 --sort-by 决定")
    ap.add_argument("--sort-by", choices=["none", "left_to_right", "area"], default=None,
                    help="人物编号排序：left_to_right=最左为0号（默认），area=最大为0号，none=SAM3原序")
    ap.add_argument("--load", default=None, help="加载降尺寸 WxH（内存不够的机器用，如 896x1184）")
    ap.add_argument("--out", default=None, help="成品下载路径（默认 ~/Downloads/scail_<时间戳>.mp4）")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--server", default=DEF_SERVER)
    ap.add_argument("--ssh-port", type=int, default=DEF_SSH_PORT)
    ap.add_argument("--json", default=str(SHARED_JSON), help="基础工作流 json（默认共享配置）")
    args = ap.parse_args()

    if len(args.images) > 5:
        ap.error("--images 最多 5 张")

    # ---------- 1. 上传输入素材（规范命名，避免和工作流文件名错位） ----------
    vid, img = Path(args.video).expanduser(), Path(args.image).expanduser()
    r_vid = f"scail_drive{vid.suffix.lower()}"
    r_img = f"scail_ref{img.suffix.lower()}"
    scp_to(args.server, args.ssh_port, str(vid), f"{SVR_INPUT}/{r_vid}")
    scp_to(args.server, args.ssh_port, str(img), f"{SVR_INPUT}/{r_img}")
    r_multis = []
    for i, m in enumerate(args.images):
        m = Path(m).expanduser()
        r = f"scail_multi_{i}{m.suffix.lower()}"
        scp_to(args.server, args.ssh_port, str(m), f"{SVR_INPUT}/{r}")
        r_multis.append(r)

    # ---------- 2. 补丁工作流 ----------
    wf = json.load(open(args.json))
    nm = {n["id"]: n for n in wf["nodes"]}

    nm[455]["widgets_values"][0] = r_img                       # 单图参考
    w471 = nm[471]["widgets_values"]                            # 驱动视频
    w471["video"] = r_vid
    w471["force_rate"] = args.fps
    w471["videopreview"]["params"].update(filename=r_vid, force_rate=args.fps)
    if args.load:
        w, h = args.load.lower().split("x")
        w471["custom_width"], w471["custom_height"] = int(w), int(h)
        w471["videopreview"]["params"]["custom_width"] = int(w)
        w471["videopreview"]["params"]["custom_height"] = int(h)
    nm[479]["widgets_values"] = [args.cap]                      # 帧数上限
    nm[457]["widgets_values"] = [args.skip]                     # 跳帧
    if args.prompt is not None:
        nm[289]["widgets_values"][2] = args.prompt              # 正面提示词
    if args.negprompt is not None:
        nm[289]["widgets_values"][3] = args.negprompt           # 负面提示词
    if args.pose_strength is not None:
        nm[576]["widgets_values"][5] = args.pose_strength       # 动作强度
    if args.ref_strength is not None:
        nm[576]["widgets_values"][6] = args.ref_strength        # 外观强度
    if args.cfg is not None:
        nm[561]["widgets_values"][1] = args.cfg                 # cfg
    if args.object_indices is not None:
        nm[490]["widgets_values"][0] = args.object_indices      # 迁移对象选择
    if args.sort_by is not None:
        nm[490]["widgets_values"][1] = args.sort_by             # 编号排序方式

    # 多图参考：给了图就解除 bypass 并填文件名，没给就保持 bypass
    for i, nid in enumerate(MULTI_NODES):
        nm[nid]["mode"] = 0 if i < len(r_multis) else 4
        if i < len(r_multis):
            nm[nid]["widgets_values"][0] = r_multis[i]
    for nid in (481, 480, 453):
        nm[nid]["mode"] = 0 if r_multis else 4

    tmp = Path("/tmp/scail_run_workflow.json")
    json.dump(wf, open(tmp, "w"), ensure_ascii=False)
    scp_to(args.server, args.ssh_port, str(tmp), "~/scail_run_workflow.json")

    # ---------- 3. 提交执行 ----------
    ssh(args.server, args.ssh_port,
        "export PATH=$HOME/.local/bin:$PATH; "
        "setsid nohup comfy run --workflow ~/scail_run_workflow.json "
        "--host 127.0.0.1 --port 8188 --wait --verbose --timeout 14400 "
        "> /tmp/scail_run.log 2>&1 </dev/null & disown; echo submitted")
    print("已提交，轮询产物...", flush=True)

    # ---------- 4. 轮询新产物并拉回 ----------
    t0 = time.time()
    marker = ssh(args.server, args.ssh_port,
                 f"ls -t {SVR_OUTGLOB}/*-audio.mp4 2>/dev/null | head -1",
                 capture=True).stdout.strip()
    while True:
        time.sleep(30)
        cur = ssh(args.server, args.ssh_port,
                  f"ls -t {SVR_OUTGLOB}/*-audio.mp4 2>/dev/null | head -1",
                  capture=True).stdout.strip()
        done = ssh(args.server, args.ssh_port,
                   "tr '\\r' '\\n' < /tmp/comfy_boot.log | grep -c 'Prompt executed in' || true",
                   capture=True).stdout.strip()
        if cur and cur != marker:
            print("产物就绪:", cur, flush=True)
            break
        alive = ssh(args.server, args.ssh_port,
                    "pgrep -f '[m]ain.py' >/dev/null && echo y || echo n",
                    capture=True).stdout.strip()
        if alive != "y":
            print("!! ComfyUI 进程消失，查 /tmp/comfy_boot.log", file=sys.stderr)
            sys.exit(2)
        print(f"  ... {int(time.time()-t0)}s (完成批次计数 {done})", flush=True)

    if not args.no_download:
        out = args.out or str(Path.home() / "Downloads" /
                              f"scail_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
        sh(["scp", "-o", "StrictHostKeyChecking=accept-new", "-P", str(args.ssh_port),
            f"{args.server}:'{cur}'", out])
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
             "-show_entries", "stream=nb_read_frames,width,height,r_frame_rate",
             "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1", out],
            capture_output=True, text=True)
        print(probe.stdout.strip())
        print("DONE ->", out)


if __name__ == "__main__":
    main()
