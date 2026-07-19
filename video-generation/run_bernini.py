#!/usr/bin/env python3
"""Bernini-R 图生视频一键调度脚本（本地驱动，远程执行）。

输入项全部参数化：
  --image     参考图 image0（角色/画面来源）
  --images    额外参考图 image1~3（可选，最多 3 张，自动解除 bypass）
  --prompt    提示词（默认沿用工作流里的 i2v 运动提示词）
  --size      画布 WxH（默认 480x640，须 16 的倍数）
  --frames    总帧数（4k+1，默认 81）/ --seconds 按时长换算
  --fps       输出帧率（默认 16）

用法示例：
  python3 run_bernini.py --image ~/Documents/Games/input.png --out ~/Downloads/out.mp4
  python3 run_bernini.py --image a.png --prompt "人物转身微笑" --seconds 15 --size 320x448
  python3 run_bernini.py --image a.png --lora-high h.safetensors --lora-low l.safetensors
  python3 run_bernini.py --image a.png --lora-high acc_h.safetensors style_h.safetensors:0.6 \
                                         --lora-low  acc_l.safetensors style_l.safetensors:0.6

依赖：ssh/scp 免密、服务器已按 runbook-bernini.md 部署（ComfyUI 在 8188，Bernini 模型就位）。
"""
import argparse, json, subprocess, sys, time, uuid
from pathlib import Path

# ---------- 服务器常量（wp08:13988 实测；换机器改这里或用参数覆盖） ----------
DEF_SERVER = "Lt2s9y@wp08.unicorn.org.cn"
DEF_SSH_PORT = 13988
SVR_INPUT = "~/comfy/ComfyUI/input"
SVR_OUTGLOB = "~/comfy/ComfyUI/output/%date:yyyy-MM-dd%"
SHARED_JSON = Path(__file__).parent / "Bernini图生视频工作流.json"

# 多图参考链路节点：image1=76, image2=25, image3=36（image0=26 恒用）
EXTRA_IMG_NODES = [76, 25, 36]


def sh(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def ssh(server, port, remote_cmd, capture=False):
    base = ["ssh", "-o", "StrictHostKeyChecking=accept-new", "-p", str(port), server, remote_cmd]
    return subprocess.run(base, check=True, capture_output=capture, text=True)


def scp_to(server, port, local, remote):
    sh(["scp", "-o", "StrictHostKeyChecking=accept-new", "-P", str(port), local, f"{server}:{remote}"])


def main():
    ap = argparse.ArgumentParser(description="Bernini-R 图生视频调度")
    ap.add_argument("--image", required=True, help="参考图 image0")
    ap.add_argument("--images", nargs="*", default=[], help="额外参考图 image1~3（最多3张）")
    ap.add_argument("--prompt", default=None, help="提示词（默认沿用工作流）")
    ap.add_argument("--negprompt", default=None, help="负面提示词（默认空=工作流默认负面）")
    ap.add_argument("--task", default="i2v", help="任务类型 i2v/r2v/t2v/v2v...（默认 i2v）")
    ap.add_argument("--frames", type=int, default=81, help="总帧数（须 4k+1，默认 81）")
    ap.add_argument("--seconds", type=float, default=None,
                    help="按时长定帧数（与 --frames 互斥；fps×秒 吸附到 4k+1）")
    ap.add_argument("--size", default="480x640", help="画布 WxH（默认 480x640，须 16 的倍数）")
    ap.add_argument("--steps", type=int, default=4, help="总步数（默认4，低步数需配加速 LoRA；底模建议 24）")
    ap.add_argument("--split", type=int, default=2, help="高低噪切分点（默认2=各半）")
    ap.add_argument("--cfg", default=None, metavar="H,L",
                    help="高低噪 cfg（默认沿用工作流 1.5,1=加速 LoRA 配套；不挂 LoRA 时建议 4,4）")
    ap.add_argument("--seed", type=int, default=None, help="种子（默认 -1 随机）")
    ap.add_argument("--fps", type=int, default=16, help="输出帧率（默认16）")
    ap.add_argument("--lora-high", nargs="+", default=[], metavar="FILE[:STRENGTH]",
                    help="高噪链 LoRA，可多个（第1个进工作流自带加载器，其余自动串联）；强度缺省 1.0")
    ap.add_argument("--lora-low", nargs="+", default=[], metavar="FILE[:STRENGTH]",
                    help="低噪链 LoRA，可多个（第1个进工作流自带加载器，其余自动串联）；强度缺省 1.0")
    ap.add_argument("--out", default=None, help="成品下载路径（默认 ~/Downloads/bernini_<时间戳>.mp4）")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--server", default=DEF_SERVER)
    ap.add_argument("--ssh-port", type=int, default=DEF_SSH_PORT)
    ap.add_argument("--json", default=str(SHARED_JSON), help="基础工作流 json（默认共享配置）")
    args = ap.parse_args()

    if len(args.images) > 3:
        ap.error("--images 最多 3 张（image1~3）")

    # ---------- 0. 帧数与画布（本地计算，不做自动适配） ----------
    if args.seconds is not None and args.frames != 81:
        ap.error("--frames 与 --seconds 二选一")
    frames = 4 * round((args.fps * args.seconds - 1) / 4) + 1 if args.seconds is not None else args.frames
    if frames < 5 or (frames - 1) % 4 != 0:
        ap.error(f"帧数须为 4k+1（Wan 时间网格），收到 {frames}")
    w16, h16 = (int(x) for x in args.size.lower().split("x"))
    if w16 % 16 or h16 % 16:
        ap.error(f"画布宽高须为 16 的倍数，收到 {w16}x{h16}")
    print(f"配置: {w16}x{h16} × {frames}f @ {args.fps}fps = {frames/args.fps:.2f}s "
          f"（{w16*h16*frames/1e6:.1f}M 像素帧）", flush=True)

    # run_id：多任务并发隔离——输入文件名/输出 prefix/轮询 glob/中转名全带它，防串号
    run_id = uuid.uuid4().hex[:8]
    print(f"run_id: {run_id}", flush=True)

    # ---------- 1. 上传输入素材 ----------
    img = Path(args.image).expanduser()
    r_img0 = f"bernini_{run_id}_ref0{img.suffix.lower()}"
    scp_to(args.server, args.ssh_port, str(img), f"{SVR_INPUT}/{r_img0}")
    r_extras = []
    for i, m in enumerate(args.images):
        m = Path(m).expanduser()
        r = f"bernini_{run_id}_ref{i+1}{m.suffix.lower()}"
        scp_to(args.server, args.ssh_port, str(m), f"{SVR_INPUT}/{r}")
        r_extras.append(r)

    # ---------- 2. 补丁工作流 ----------
    wf = json.load(open(args.json))
    nm = {n["id"]: n for n in wf["nodes"]}

    # 虚拟节点兼容（共享 json 是前端原版，这里动态打补丁，不改本地文件）
    nm[31]["type"] = "PrimitiveInt"
    nm[31]["widgets_values"] = [frames]                   # 直接给总帧数
    nm[30]["widgets_values"] = ["a"]                      # SimpleMath 直通（原为 a*16+1）
    wf["nodes"] = [n for n in wf["nodes"] if n.get("type") != "Label (rgthree)"]
    for n in wf["nodes"]:
        if n.get("type") == "Fast Groups Bypasser (rgthree)":
            n["type"] = "FastGroupsBypassSwitch"
    links = wf.get("links", [])
    label_ids = {65, 66, 67, 69}
    wf["links"] = [l for l in links if l[1] not in label_ids and l[3] not in label_ids]

    # 图片输入
    nm[26]["mode"] = 0
    nm[26]["widgets_values"] = [r_img0, "image"]
    for i, nid in enumerate(EXTRA_IMG_NODES):
        nm[nid]["mode"] = 0 if i < len(r_extras) else 4
        if i < len(r_extras):
            nm[nid]["widgets_values"] = [r_extras[i], "image"]

    nm[34]["widgets_values"] = [w16]
    nm[35]["widgets_values"] = [h16]

    # BerniniStudio 本体
    w27 = nm[27]["widgets_values"]
    w27[4] = args.task
    if args.prompt is not None:
        w27[5] = args.prompt
    if args.negprompt is not None:
        w27[6] = args.negprompt
    w27[9] = False                                       # auto_enhance：服务器无 Ollama

    # LoRA：--lora-high/--lora-low 各收多个 文件[:强度]，默认底模（7/10 bypass）
    def parse_lora_specs(specs):
        out = []
        for s in specs:
            head, sep, tail = s.rpartition(":")
            out.append((head, float(tail)) if sep and tail.replace(".", "", 1).isdigit() else (s, 1.0))
        return out
    high_loras = parse_lora_specs(args.lora_high)
    low_loras = parse_lora_specs(args.lora_low)
    for nid, loras in ((7, high_loras), (10, low_loras)):
        nm[nid]["mode"] = 0 if loras else 4
        if loras:
            nm[nid]["widgets_values"] = list(loras[0])
    if not (high_loras or low_loras) and args.steps <= 6:
        print("!! 提示: 未挂 LoRA 且步数 ≤6，底模低步数出片质量差；"
              "建议 --steps 24 --cfg 4,4 或挂加速 LoRA", flush=True)
    for name, _ in high_loras + low_loras:
        ok = ssh(args.server, args.ssh_port,
                 f"test -f ~/comfy/ComfyUI/models/loras/'{name}' && echo y || echo n",
                 capture=True).stdout.strip()
        if ok != "y":
            ap.error(f"服务器 loras/ 下找不到 {name}，先 scp 上传")

    # 每条链第 2 个起：在 7→SetNode46 / 10→SetNode48 之间插 LoraLoaderModelOnly 链节点
    extra = [(consumer, name, strength)
             for consumer, loras in ((46, high_loras[1:]), (48, low_loras[1:]))
             for name, strength in loras]
    if extra:
        max_id = max(n["id"] for n in wf["nodes"])
        max_link = max(l[0] for l in wf["links"])
        tpl = nm[7]
        tails = {46: [7, 97], 48: [10, 98]}   # consumer_id: [当前链尾节点, 链尾 link]
        for consumer, name, strength in extra:
            tail_node, tail_link = tails[consumer]
            max_id += 1
            max_link += 1
            xn = json.loads(json.dumps(tpl))
            xn["id"] = max_id
            xn["mode"] = 0
            xn["widgets_values"] = [name, strength]
            xn["inputs"] = [{"name": "model", "type": "MODEL", "link": max_link}]
            xn["outputs"] = [{"name": "MODEL", "type": "MODEL",
                              "links": [tail_link], "slot_index": 0}]
            xn["pos"] = [tpl["pos"][0] + 280, tpl["pos"][1]]
            for l in wf["links"]:
                if l[0] == tail_link:
                    l[1] = max_id                     # consumer 改从 xn 收
            wf["links"].append([max_link, tail_node, 0, max_id, 0, "MODEL"])
            for o in nm[tail_node]["outputs"]:
                if o.get("links"):
                    o["links"] = [max_link if x == tail_link else x for x in o["links"]]
            wf["nodes"].append(xn)
            nm[max_id] = xn
            tails[consumer] = [max_id, tail_link]
            print(f"串联 LoRA: {name} x{strength} → 节点 {max_id}（链 {tail_node}→{consumer}）", flush=True)

    # 采样参数
    nm[13]["widgets_values"] = ["simple", args.steps, 1]
    nm[14]["widgets_values"] = [args.split]
    if args.cfg:
        cfg_h, cfg_l = (float(x) for x in args.cfg.split(","))
        nm[15]["widgets_values"][3] = cfg_h    # High Sampler cfg（widgets[3]）
        nm[16]["widgets_values"][3] = cfg_l    # Low Sampler cfg
    if args.seed is not None:
        nm[29]["widgets_values"][0] = args.seed
    nm[22]["widgets_values"]["frame_rate"] = args.fps
    nm[22]["widgets_values"]["filename_prefix"] = \
        nm[22]["widgets_values"]["filename_prefix"] + "_" + run_id  # 输出文件名带 run_id

    tmp = Path("/tmp/run_bernini_workflow.json")
    json.dump(wf, open(tmp, "w"), ensure_ascii=False)
    scp_to(args.server, args.ssh_port, str(tmp), "~/run_bernini_workflow.json")

    # ---------- 3. 提交执行 ----------
    ssh(args.server, args.ssh_port,
        "export PATH=$HOME/.local/bin:$PATH; "
        "setsid nohup comfy run --workflow ~/run_bernini_workflow.json "
        "--host 127.0.0.1 --port 8188 --wait --verbose --timeout 7200 "
        "> /tmp/run_bernini.log 2>&1 </dev/null & disown; echo submitted")
    print("已提交，轮询产物...", flush=True)

    # ---------- 4. 轮询新产物并拉回 ----------
    t0 = time.time()
    out_glob = f"{SVR_OUTGLOB}/*Bernini_{run_id}_*.mp4"
    marker = ssh(args.server, args.ssh_port,
                 f"ls -t {out_glob} 2>/dev/null | head -1",
                 capture=True).stdout.strip()
    while True:
        time.sleep(20)
        cur = ssh(args.server, args.ssh_port,
                  f"ls -t {out_glob} 2>/dev/null | head -1",
                  capture=True).stdout.strip()
        if cur and cur != marker:
            print("产物就绪:", cur, flush=True)
            break
        alive = ssh(args.server, args.ssh_port,
                    "pgrep -f '[m]ain.py' >/dev/null && echo y || echo n",
                    capture=True).stdout.strip()
        if alive != "y":
            err = ssh(args.server, args.ssh_port,
                      "tail -5 /tmp/run_bernini.log", capture=True).stdout
            print("!! ComfyUI 进程消失\n", err, file=sys.stderr)
            sys.exit(2)
        # 客户端已退出（校验失败/报错）但无产物 → 及时止损
        done = ssh(args.server, args.ssh_port,
                   "grep -qE '\"ok\": false|Traceback' /tmp/run_bernini.log 2>/dev/null && echo FAIL || echo OK",
                   capture=True).stdout.strip()
        if done == "FAIL":
            err = ssh(args.server, args.ssh_port,
                      "tail -8 /tmp/run_bernini.log", capture=True).stdout
            print("!! 执行失败\n", err, file=sys.stderr)
            sys.exit(3)
        print(f"  ... {int(time.time()-t0)}s", flush=True)

    if not args.no_download:
        out = args.out or str(Path.home() / "Downloads" /
                              f"bernini_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
        # 远端路径含 % 和 :，SFTP 模式下引号不被解释 → 先 cp 到 /tmp 简单名再拉
        tmp_remote = f"/tmp/bernini_out_{run_id}.mp4"
        ssh(args.server, args.ssh_port, f"cp '{cur}' {tmp_remote}")
        sh(["scp", "-o", "StrictHostKeyChecking=accept-new", "-P", str(args.ssh_port),
            f"{args.server}:{tmp_remote}", out])
        ssh(args.server, args.ssh_port,
            f"rm -f {tmp_remote} {SVR_INPUT}/bernini_{run_id}_ref*")  # 清本场输入/中转
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
             "-show_entries", "stream=nb_read_frames,width,height,r_frame_rate",
             "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1", out],
            capture_output=True, text=True)
        print(probe.stdout.strip())
        print("DONE ->", out)


if __name__ == "__main__":
    main()
