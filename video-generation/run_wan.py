#!/usr/bin/env python3
"""
run_wan.py — Wan2.2 官方 I2V（图生视频）一键跑（服务器 wp08:13988 / 4090 24G）

底图：Comfy-Org 官方极简模板（Wan图生视频工作流.json；曾用 Kenpechi SVI v3.5，
其 FMLF WanAdvancedI2V 节点在本服务器产出混沌，已弃用，见 runbook-wan.md）。

  python3 run_wan.py --image a.png --prompt "……" --seconds 5
  python3 run_wan.py --image a.png --lora-high H1.safetensors:0.6 H2.safetensors \
                     --lora-low L1.safetensors:0.6 L2.safetensors --base gguf

要点：
  - 图结构：UNETLoader×2 →[LoRA槽 126/127 + 可串联插入]→ ModelSamplingSD3(shift5)
    → KSamplerAdvanced×2（高噪 0→split / 低噪 split→end，euler+simple）
    → WanImageToVideo(36ch 官方 i2v 条件) → VAEDecode → CreateVideo → SaveVideo
  - 关键节点（子图 d2ac71a3 内）：CLIP 105 / VAE 106 / 正 107 / 负 125 /
    UNET 122,123 / LoRA 126,127 / shift 109,124 / 采样 110,111 / i2v 128 / fps 117
  - 顶层：LoadImage 97 / SaveVideo 108（输出 h264 mp4，QuickTime 可播）
  - 挂 LoRA 用 --base gguf（fp8_scaled + LoRA 在 lowvram 下重量化必 OOM）；
    不挂 LoRA 时 fp8 更快更小，二选一。
"""
import argparse, json, os, random, subprocess, sys, time, uuid
from pathlib import Path

SRV, PORT = "Lt2s9y@wp08.unicorn.org.cn", 13988
SSH_OPTS = ["-o", "StrictHostKeyChecking=accept-new"]  # TOFU：首连收指纹并记录；绝不用 no
SHARED_JSON = Path(__file__).parent / "Wan图生视频工作流.json"
SVR_INPUT = "~/comfy/ComfyUI/input"
SVR_OUTPUT = "~/comfy/ComfyUI/output"

UMT5 = "umt5-xxl-enc-fp8_e4m3fn.safetensors"  # 服务器上的 umt5（fp16 内容沿用旧名）
VAE = "wan_2.1_vae.safetensors"
FP8_HIGH = "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
FP8_LOW = "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
GGUF_HIGH = "wan2.2_i2v_high_noise_14B_Q8_0.gguf"
GGUF_LOW = "wan2.2_i2v_low_noise_14B_Q8_0.gguf"

# 子图内节点
N_CLIP, N_VAE = 105, 106
N_POS, N_NEG = 107, 125
N_UNET_H, N_UNET_L = 122, 123
N_LORA_H, N_LORA_L = 126, 127
N_SAMP_H, N_SAMP_L = 110, 111
N_WAN_I2V = 128
N_FPS = 117
# 顶层
N_IMG, N_SAVE = 97, 108


def sh(cmd):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def ssh(server, port, cmd, capture=False, retries=1):
    """retries>1 时仅在 exit 255（连接被远端断开，wp08 新机偶发）下重试；
    只用于只读轮询调用，提交类写调用保持 retries=1 防重复提交。"""
    for attempt in range(1, retries + 1):
        try:
            return subprocess.run(
                ["ssh", *SSH_OPTS, "-o", "ConnectTimeout=15",
                 "-p", str(port), server, cmd],
                capture_output=capture, text=True, check=True)
        except subprocess.CalledProcessError as e:
            if e.returncode != 255 or attempt == retries:
                raise
            print(f"!! ssh 255（第{attempt}/{retries}次），5s 后重试",
                  file=sys.stderr, flush=True)
            time.sleep(5)


def scp_to(server, port, local, remote):
    sh(["scp", *SSH_OPTS, "-P", str(port), local, f"{server}:{remote}"])


def parse_lora(specs):
    """FILE[:STRENGTH]... → [(file, strength)]，缺省强度 1.0"""
    out = []
    for s in specs:
        f, _, st = s.rpartition(":")
        if f and st.replace(".", "", 1).isdigit():
            out.append((f, float(st)))
        else:
            out.append((s, 1.0))
    return out


def sub_nodes(wf):
    sg = wf["definitions"]["subgraphs"][0]
    return sg, {n["id"]: n for n in sg["nodes"]}, {l["id"]: l for l in sg["links"]}


def chain_lora(sg, sn, after_id, before_id, lora_file, strength, new_nid):
    """在 after_id → before_id 的 MODEL 边上插入一个 LoraLoaderModelOnly。
    注意：必须直接改 sg["links"] 列表（自建字典是副本，改了不生效）。"""
    links = sg["links"]
    new_lid = max(l["id"] for l in links) + 1
    tgt = None
    for n_in in sn[before_id].get("inputs", []):
        lk = n_in.get("link")
        if lk is None:
            continue
        d = next((l for l in links if l["id"] == lk), None)
        if d and d["origin_id"] == after_id:
            tgt = d
            break
    assert tgt is not None, f"找不到 {after_id}→{before_id} 的 MODEL 边"
    links.remove(tgt)                             # 摘除 after→before 旧边
    for o in sn[after_id].get("outputs", []):     # after 输出：旧边换新边
        if o.get("links") and tgt["id"] in o["links"]:
            o["links"].remove(tgt["id"])
            o["links"].append(new_lid)
    node = {"id": new_nid, "type": "LoraLoaderModelOnly",
            "pos": [sn[after_id]["pos"][0] + 200, sn[after_id]["pos"][1] + 80],
            "size": [315, 106], "flags": {}, "order": 0, "mode": 0,
            "inputs": [{"name": "model", "type": "MODEL", "link": new_lid}],
            "outputs": [{"name": "MODEL", "type": "MODEL", "links": [new_lid + 1],
                         "slot_index": 0}],
            "properties": {"Node name for S&R": "LoraLoaderModelOnly"},
            "widgets_values": [lora_file, strength]}
    sg["nodes"].append(node)
    sn[new_nid] = node
    links.append({"id": new_lid, "origin_id": after_id,
                  "origin_slot": tgt["origin_slot"], "target_id": new_nid,
                  "target_slot": 0, "type": "MODEL"})
    links.append({"id": new_lid + 1, "origin_id": new_nid, "origin_slot": 0,
                  "target_id": before_id, "target_slot": tgt["target_slot"],
                  "type": "MODEL"})
    for n_in in sn[before_id].get("inputs", []):
        if n_in.get("link") == tgt["id"]:
            n_in["link"] = new_lid + 1
    return new_nid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="参考图（首帧）")
    ap.add_argument("--prompt", default=None, help="提示词（默认沿用模板）")
    ap.add_argument("--negprompt", default=None, help="负面提示词（默认沿用模板）")
    ap.add_argument("--seconds", type=float, default=3, help="时长秒（默认3；帧数=秒*fps+1）")
    ap.add_argument("--size", default="480x640", help="画布 WxH（默认 480x640，须 16 的倍数）")
    ap.add_argument("--steps", type=int, default=12, help="总步数（默认12；无加速 LoRA 建议 12~24）")
    ap.add_argument("--split", type=int, default=6, help="高低噪切分点（默认6=步数一半）")
    ap.add_argument("--cfg", default="4,4", metavar="H,L", help="高噪,低噪 cfg（默认 4,4）")
    ap.add_argument("--seed", type=int, default=None, help="种子（默认随机）")
    ap.add_argument("--fps", type=int, default=16, help="帧率（默认16）")
    ap.add_argument("--lora-high", nargs="+", default=[], metavar="FILE[:STRENGTH]",
                    help="高噪链 LoRA，可多个；第 2 个起自动串联插节点")
    ap.add_argument("--lora-low", nargs="+", default=[], metavar="FILE[:STRENGTH]",
                    help="低噪链 LoRA，可多个；第 2 个起自动串联插节点")
    ap.add_argument("--base", choices=["fp8", "gguf"], default="fp8",
                    help="底模：fp8=fp8_scaled（默认，不挂 LoRA 时用）；gguf=Q8_0（挂 LoRA 必选，"
                         "fp8+LoRA 在 lowvram 下重量化必 OOM）")
    ap.add_argument("--unet-dtype", default="default", metavar="DTYPE",
                    help="仅 base=fp8：UNETLoader 的 weight_dtype")
    ap.add_argument("--out", default=None, help="本地输出路径（默认 ~/Downloads/wan22_时间戳.mp4）")
    ap.add_argument("--no-download", action="store_true", help="只跑不拉回")
    ap.add_argument("--server", default=SRV)
    ap.add_argument("--ssh-port", type=int, default=PORT)
    args = ap.parse_args()

    run_id = uuid.uuid4().hex[:8]
    seed = args.seed if args.seed is not None else random.randint(0, 2**53 - 1)
    w16, h16 = (int(x) for x in args.size.lower().split("x"))
    assert w16 % 16 == 0 and h16 % 16 == 0, "--size 须 16 的倍数"
    frames = round(args.seconds * args.fps) + 1
    frames = (frames - 1) // 4 * 4 + 1
    cfg_h, cfg_l = (float(x) for x in args.cfg.split(","))
    lora_h, lora_l = parse_lora(args.lora_high), parse_lora(args.lora_low)
    if (lora_h or lora_l) and args.base != "gguf":
        print("!! 挂了 LoRA 但 --base 不是 gguf：fp8+LoRA 在此服务器必 OOM，已自动切 gguf",
              file=sys.stderr)
        args.base = "gguf"

    print(f"run_id={run_id} seed={seed} 画布={w16}x{h16} 帧数={frames}@{args.fps}fps"
          f"（{w16*h16*frames/1e6:.1f}M 像素帧） steps={args.steps}/{args.split}"
          f" cfg={cfg_h},{cfg_l} base={args.base}", flush=True)

    # ---------- 1. 上传参考图 ----------
    ext_img = Path(args.image).suffix or ".png"
    remote_img = f"{SVR_INPUT}/wan_{run_id}_ref{ext_img}"
    scp_to(args.server, args.ssh_port, args.image, remote_img)

    # ---------- 2. 补丁工作流 ----------
    wf = json.load(open(SHARED_JSON))
    sg, sn, sl = sub_nodes(wf)
    nm = {n["id"]: n for n in wf["nodes"]}

    nm[N_IMG]["widgets_values"][0] = f"wan_{run_id}_ref{ext_img}"
    nm[N_SAVE]["widgets_values"][0] = f"Wan22_{run_id}"

    if args.prompt:
        sn[N_POS]["widgets_values"][0] = args.prompt
    if args.negprompt:
        sn[N_NEG]["widgets_values"][0] = args.negprompt
    sn[N_CLIP]["widgets_values"][0] = UMT5
    sn[N_VAE]["widgets_values"][0] = VAE

    # 底模
    if args.base == "gguf":
        for nid, gf in ((N_UNET_H, GGUF_HIGH), (N_UNET_L, GGUF_LOW)):
            sn[nid]["type"] = "UnetLoaderGGUF"
            sn[nid]["widgets_values"] = [gf]
    else:
        for nid, ff in ((N_UNET_H, FP8_HIGH), (N_UNET_L, FP8_LOW)):
            sn[nid]["widgets_values"] = [ff, args.unet_dtype]

    # 采样器（高：0→split 加噪；低：split→end 不加噪，接力高噪潜变量）
    sn[N_SAMP_H]["widgets_values"] = ["enable", seed, "fixed", args.steps, cfg_h,
                                      "euler", "simple", 0, args.split, "enable"]
    sn[N_SAMP_L]["widgets_values"] = ["disable", seed, "fixed", args.steps, cfg_l,
                                      "euler", "simple", args.split, 999, "disable"]

    # i2v 条件与帧率
    sn[N_WAN_I2V]["widgets_values"] = [w16, h16, frames, 1]
    sn[N_FPS]["widgets_values"][0] = args.fps

    # LoRA：第 1 只进内置槽（126 高 / 127 低），第 2 只起串联插入
    for nid, shift_nid, loras, tag in (
            (N_LORA_H, 109, lora_h, "高"), (N_LORA_L, 124, lora_l, "低")):
        if not loras:
            sn[nid]["mode"] = 4
            continue
        sn[nid]["mode"] = 0
        f, st = loras[0]
        sn[nid]["widgets_values"] = [f, st]
        print(f"LoRA[{tag}][{nid}]: {f} x{st}", flush=True)
        prev = nid
        for i, (f2, st2) in enumerate(loras[1:], start=1):
            new_id = 500 + (nid % 100) * 10 + i
            chain_lora(sg, sn, prev, shift_nid, f2, st2, new_id)
            print(f"LoRA[{tag}][{new_id}]: {f2} x{st2}（串联）", flush=True)
            prev = new_id

    tmp = Path("/tmp/run_wan_workflow.json")
    json.dump(wf, open(tmp, "w"), ensure_ascii=False)
    scp_to(args.server, args.ssh_port, str(tmp), "~/run_wan_workflow.json")

    # ---------- 3. 提交执行 ----------
    ssh(args.server, args.ssh_port,
        "export PATH=$HOME/.local/bin:$PATH; "
        "setsid nohup comfy run --workflow ~/run_wan_workflow.json "
        "--host 127.0.0.1 --port 8188 --wait --verbose --timeout 7200 "
        "> /tmp/run_wan.log 2>&1 </dev/null & disown; echo submitted")
    print("已提交，轮询产物...", flush=True)

    # ---------- 4. 轮询产物并拉回 ----------
    # 完成信号 = comfy run 进程退出（文件出现≠写完，h264 的 moov 最后才落盘，
    # 按文件出现就拉会拉到截断 mp4）。失败信号 = 日志出现错误信封/Traceback。
    t0 = time.time()
    out_glob = f"{SVR_OUTPUT}/Wan22_{run_id}*"
    while True:
        time.sleep(20)
        done = ssh(args.server, args.ssh_port,
                   "grep -qE '\"ok\": false|Traceback' /tmp/run_wan.log 2>/dev/null && echo FAIL || echo OK",
                   capture=True, retries=6).stdout.strip()
        if done == "FAIL":
            err = ssh(args.server, args.ssh_port,
                      "tail -8 /tmp/run_wan.log", capture=True).stdout
            print("!! 执行失败\n", err, file=sys.stderr)
            sys.exit(3)
        running = ssh(args.server, args.ssh_port,
                      "pgrep -f 'comfy r[u]n --workflow' >/dev/null && echo y || echo n",
                      capture=True, retries=6).stdout.strip()
        if running != "y":
            break  # comfy run 已退出 = 流程结束（成败上面已判）
        alive = ssh(args.server, args.ssh_port,
                    "pgrep -f '[m]ain.py' >/dev/null && echo y || echo n",
                    capture=True, retries=6).stdout.strip()
        if alive != "y":
            err = ssh(args.server, args.ssh_port,
                      "tail -5 /tmp/run_wan.log", capture=True).stdout
            print("!! ComfyUI 进程消失\n", err, file=sys.stderr)
            sys.exit(2)
        if time.time() - t0 > 7200:
            print("!! 轮询超时（2h）", file=sys.stderr)
            sys.exit(5)
        print(f"  ... {int(time.time()-t0)}s", flush=True)

    cur = ssh(args.server, args.ssh_port,
              f"ls -t {out_glob} 2>/dev/null | head -1",
              capture=True, retries=6).stdout.strip()
    if not cur:
        err = ssh(args.server, args.ssh_port,
                  "tail -8 /tmp/run_wan.log", capture=True).stdout
        print("!! 流程结束但找不到产物\n", err, file=sys.stderr)
        sys.exit(4)
    print("产物就绪:", cur, flush=True)

    if not args.no_download:
        out = args.out or str(Path.home() / "Downloads" /
                              f"wan22_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
        tmp_remote = f"/tmp/wan_out_{run_id}{os.path.splitext(cur)[1] or '.mp4'}"
        for attempt in (1, 2):
            ssh(args.server, args.ssh_port, f"cp '{cur}' {tmp_remote}")
            sh(["scp", *SSH_OPTS, "-P", str(args.ssh_port),
                f"{args.server}:{tmp_remote}", out])
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                 "-show_entries", "stream=nb_read_frames,width,height,r_frame_rate",
                 "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1", out],
                capture_output=True, text=True)
            if probe.returncode == 0 and "nb_read_frames=" in probe.stdout \
                    and not probe.stdout.strip().endswith("nb_read_frames=0"):
                break
            print(f"!! 产物校验失败（第{attempt}次），5s 后重拉\n"
                  f"{probe.stderr.strip()[:300]}", flush=True)
            time.sleep(5)
        else:
            print("!! 产物仍不可读，服务器原件保留在 " + cur, file=sys.stderr)
            sys.exit(4)
        ssh(args.server, args.ssh_port,
            f"rm -f {tmp_remote} {SVR_INPUT}/wan_{run_id}_ref*")
        print(probe.stdout.strip())
        print("DONE ->", out)
    else:
        print("DONE (no-download)")


if __name__ == "__main__":
    main()
