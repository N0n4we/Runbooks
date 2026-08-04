#!/usr/bin/env python3
"""
run-h3.py — MiniMax H3 全模态视频生成一键跑（服务器 wp08:33307 / 3090 24G）

底图：Comfy-Org 官方模板两枚——H3文生视频工作流.json（fl2va；i2v 由脚本动态
接首/末帧图覆盖）与 H3参考生视频工作流.json（ref2va）。
ComfyUI ≥0.30.0 原生节点，无需任何第三方节点包。暂不支持 LoRA。

模态自动选择（互斥）：
  纯文本          → T2V（fl2va 底模）
  --image/--last-image → I2V 首/末帧（fl2va 底模）
  --ref-image/--ref-video/--ref-audio → R2V（ref2va 底模）

  python3 run-h3.py --prompt "……"
  python3 run-h3.py --prompt "……" --image first.png --last-image last.png
  python3 run-h3.py --prompt "让 <Picture 1> 的角色在 <Video 1> 的场景里……" \
      --ref-image role.png --ref-video scene.mp4 --ref-audio voice.mp3

参考标签约定（R2V 提示词里引用，1 起始）：
  <Picture i>  参考图，按 --ref-image 顺序
  <Video k>    参考视频，按 --ref-video 顺序
  <Audio j>    先是各参考视频自带音轨（按视频顺序），再是 --ref-audio 独立音频

要点：
  - 帧数网格 17k+5 @24fps（5s=124 帧）；训练范围 124~362 帧（≈5~15s）
  - 画布 32 倍数；原生 768 短边、面积上限 768×1344（≈0.4MP 模板默认快速预览）
  - BasicGuider=cfg1 无负面词；采样 res_multistep + simple 20 步（模板默认）
  - R2V 上限：9 图 / 3 视频（各带音轨）/ 3 独立音频
  - --shift-video/--shift-audio：插入 MiniMaxH3SigmaShift 节点（接管 UNET 全部
    MODEL 出边，调度器与引导器用同一份 patched model）调视频/音频流 flow shift
"""
import argparse, json, os, random, subprocess, sys, time, uuid
from pathlib import Path

SRV, PORT = "RxxrJp@wp08.unicorn.org.cn", 13054  # 主目标机（4090 24G，int4 套路）
SSH_OPTS = ["-o", "StrictHostKeyChecking=accept-new"]  # TOFU：首连收指纹并记录；绝不用 no
JSON_DIR = Path(__file__).parent
SVR_INPUT = "~/comfy/ComfyUI/input"
SVR_OUTPUT = "~/comfy/ComfyUI/output"

UNET_FL2VA = "MiniMax_H3_FL2VA_pruned_int4_convrot.safetensors"
UNET_REF2VA = "MiniMax_H3_Ref2VA_pruned_int4_convrot.safetensors"
CLIP_QWEN = "qwen3vl_32b_minimax_h3_int4_convrot.safetensors"  # int4 套路；nvfp4 是 Blackwell 专用不可用
VAE_VIDEO = "minimax_h3_video_vae_fp16.safetensors"
VAE_AUDIO = "minimax_h3_audio_vae_fp32.safetensors"

ASPECTS = ["1:1 (Square)", "2:3 (Portrait Photo)", "3:2 (Photo)", "3:4 (Portrait Standard)",
           "4:3 (Standard)", "9:16 (Portrait Widescreen)", "16:9 (Widescreen)", "21:9 (Ultrawide)"]


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


def snap_frames(n):
    """H3 帧数网格 17k+5，向上吸附，下限 5。"""
    n = max(5, n)
    while n % 17 != 5:
        n += 1
    return n


def insert_sigma_shift(g, unet_id, shift_video, shift_audio):
    """在 UNETLoader 后插 MiniMaxH3SigmaShift，并接管其全部 MODEL 出边。
    必须同时喂 BasicGuider 和 BasicScheduler——shift 改变 model_sampling，
    调度器要从同一份 patched model 算 sigma，否则调度表与 DiT 内部不一致。"""
    pos = g.nm[unet_id]["pos"]
    sid = g.add_node("MiniMaxH3SigmaShift", [shift_video, shift_audio],
                     [("MODEL", "MODEL")], [pos[0] + 350, pos[1] + 40])
    targets = [(g._l(l, "id"), g._l(l, "target_id"), g._l(l, "target_slot"))
               for l in list(g.links)
               if g._l(l, "origin_id") == unet_id and g._l(l, "type") == "MODEL"]
    for old_lid, tgt, tslot in targets:
        g.cut_link(old_lid)
    g.link(unet_id, 0, sid, 0, "MODEL", name="model")
    for _, tgt, tslot in targets:
        g.link(sid, 0, tgt, tslot, "MODEL")
    return sid


class Graph:
    """前端格式工作流的补丁助手：节点/链接索引 + 增删。
    注意 Comfy-Org 模板 links 是混合格式：顶层为列表 [id,org,oslot,tgt,tslot,type]，
    子图定义为 dict。读改写都按本作用域的原生格式走。"""

    def __init__(self, wf, scope_nodes, scope_links):
        self.wf = wf
        self.nodes = scope_nodes          # list（直接改，勿换引用）
        self.links = scope_links          # list（直接改，勿换引用）
        self.nm = {n["id"]: n for n in scope_nodes}
        self.fmt = "dict" if (scope_links and isinstance(scope_links[0], dict)) else "list"

    def _l(self, l, key):
        """link 取字段：id/origin_id/origin_slot/target_id/target_slot/type"""
        if isinstance(l, dict):
            return l[key]
        idx = {"id": 0, "origin_id": 1, "origin_slot": 2,
               "target_id": 3, "target_slot": 4, "type": 5}[key]
        return l[idx]

    def _mk_link(self, lid, org, oslot, tgt, tslot, ltype):
        if self.fmt == "dict":
            return {"id": lid, "origin_id": org, "origin_slot": oslot,
                    "target_id": tgt, "target_slot": tslot, "type": ltype}
        return [lid, org, oslot, tgt, tslot, ltype]

    def nid(self, start=900):
        used = set(self.nm)
        i = start
        while i in used:
            i += 1
        return i

    def lid(self):
        return max((self._l(l, "id") for l in self.links), default=0) + 1

    def input_slot(self, node_id, name):
        for idx, inp in enumerate(self.nm[node_id].get("inputs", [])):
            if inp["name"] == name:
                return idx, inp
        return None, None

    def cut_link(self, link_id):
        """摘一条边：links 列表 + 两端节点的 link/links 引用一起清。"""
        d = next((l for l in self.links if self._l(l, "id") == link_id), None)
        if d is None:
            return
        self.links.remove(d)
        org = self.nm.get(self._l(d, "origin_id"))
        if org:
            for o in org.get("outputs", []):
                if o.get("links") and link_id in o["links"]:
                    o["links"].remove(link_id)
        tgt = self.nm.get(self._l(d, "target_id"))
        if tgt:
            for i in tgt.get("inputs", []):
                if i.get("link") == link_id:
                    i["link"] = None

    def link(self, org_id, org_slot, tgt_id, tgt_slot, ltype, name=None,
             shape=None, label=None):
        """新建一条边；name 定位已有槽（动态槽缺失时补建），否则按 tgt_slot 落已有槽。返回 link id。"""
        lid = self.lid()
        tgt = self.nm[tgt_id]
        if name is not None:
            idx, inp = self.input_slot(tgt_id, name)
        else:
            idx = tgt_slot
            ins = tgt.get("inputs", [])
            inp = ins[idx] if idx is not None and idx < len(ins) else None
        if inp is None:
            assert name is not None, f"link: 节点 {tgt_id} 槽位 {tgt_slot} 不存在且未给 name"
            entry = {"name": name, "type": ltype, "link": lid}
            if label:
                entry["label"] = label
            if shape is not None:
                entry["shape"] = shape
            tgt.setdefault("inputs", []).append(entry)
            idx = len(tgt["inputs"]) - 1
        else:
            if inp.get("link") is not None:
                self.cut_link(inp["link"])   # 重连语义：先剪旧边
            inp["link"] = lid
        org = self.nm[org_id]
        outs = org.setdefault("outputs", [])
        while len(outs) <= org_slot:
            outs.append({"name": "", "type": ltype, "links": []})
        outs[org_slot].setdefault("links", [])
        if outs[org_slot]["links"] is None:
            outs[org_slot]["links"] = []
        outs[org_slot]["links"].append(lid)
        self.links.append(self._mk_link(
            lid, org_id, org_slot, tgt_id,
            idx if idx is not None else tgt_slot, ltype))
        return lid

    def add_node(self, ntype, widgets, outputs, pos, title=None):
        nid = self.nid()
        node = {"id": nid, "type": ntype, "pos": list(pos), "size": [315, 106],
                "flags": {}, "order": 0, "mode": 0, "inputs": [],
                "outputs": [{"name": o, "type": t, "links": []} for o, t in outputs],
                "properties": {"Node name for S&R": ntype},
                "widgets_values": widgets}
        if title:
            node["title"] = title
        self.nodes.append(node)
        self.nm[nid] = node
        return nid

    def drop_node(self, nid):
        """删节点及其全部相连边（本脚本只删纯输入源节点）。"""
        for l in list(self.links):
            if self._l(l, "origin_id") == nid or self._l(l, "target_id") == nid:
                self.cut_link(self._l(l, "id"))
        node = self.nm.pop(nid, None)
        if node in self.nodes:
            self.nodes.remove(node)


def upload_inputs(args, run_id, files):
    """[(local_path, tag)] → {tag: remote_basename}"""
    names = {}
    for local, tag in files:
        ext = Path(local).suffix or ".bin"
        base = f"h3_{run_id}_{tag}{ext}"
        scp_to(args.server, args.ssh_port, local, f"{SVR_INPUT}/{base}")
        names[tag] = base
    return names


def patch_common_fl2va(g_top, g_sg, args, seed, frames, frames_direct):
    """t2v/i2v 共享的子图补丁。g_top=顶层图，g_sg=子图定义图。"""
    inst = g_top.nm[105]           # 子图实例（promoted widgets 权威）
    sn = g_sg.nm                   # 子图内部节点
    # 底模/编码器/VAE（实例 promoted widgets + 内部节点双写，防转换器口径不一）
    inst["widgets_values"][5] = UNET_FL2VA
    inst["widgets_values"][6] = CLIP_QWEN      # nvfp4 → int8（3090）
    inst["widgets_values"][7] = VAE_VIDEO
    inst["widgets_values"][8] = VAE_AUDIO
    sn[6]["widgets_values"][0] = UNET_FL2VA
    sn[13]["widgets_values"][0] = CLIP_QWEN
    sn[11]["widgets_values"][0] = VAE_VIDEO
    sn[24]["widgets_values"][0] = VAE_AUDIO
    # 提示词 / 种子
    inst["widgets_values"][0] = args.prompt
    sn[104]["widgets_values"][0] = args.prompt
    inst["widgets_values"][4] = seed
    sn[15]["widgets_values"] = [seed, "fixed"]
    # 采样
    sn[17]["widgets_values"] = [args.sampler]
    sn[9]["widgets_values"] = [args.scheduler, args.steps, 1]
    # 时长：--frames 直写 length（剪 107→104 内部边），否则走 PrimitiveFloat 秒链
    if frames_direct:
        idx, inp = g_sg.input_slot(104, "length")
        if inp and inp.get("link") is not None:
            g_sg.cut_link(inp["link"])
        sn[104]["widgets_values"][3] = frames
    else:
        inst["widgets_values"][3] = args.seconds
        sn[111]["widgets_values"][0] = args.seconds


def patch_size(g, tgt_id, w_name, h_name, args, widgets_node=None, wv_w=1, wv_h=2):
    """--size：PrimitiveInt×2 重接线；否则补丁 ResolutionSelector(115)。"""
    if args.size:
        w, h = (int(x) for x in args.size.lower().split("x"))
        assert w % 32 == 0 and h % 32 == 0, "--size 须 32 的倍数"
        for name, val in ((w_name, w), (h_name, h)):
            idx, inp = g.input_slot(tgt_id, name)
            if inp and inp.get("link") is not None:
                g.cut_link(inp["link"])
            pos = g.nm[tgt_id]["pos"]
            nid = g.add_node("PrimitiveInt", [val], [("INT", "INT")],
                             [pos[0] - 420, pos[1] + (0 if name == w_name else 60)])
            g.link(nid, 0, tgt_id, idx, "INT")
        if widgets_node is not None:
            widgets_node["widgets_values"][wv_w] = w
            widgets_node["widgets_values"][wv_h] = h
        return w, h
    else:
        g.nm[115]["widgets_values"] = [args.aspect, args.megapixels, 32]
        return None, None


def build_workflow(args, run_id, seed, up):
    """按模态选模板并补丁，返回 (wf, mode, frames)。"""
    if args.frames:
        frames = snap_frames(args.frames)
    else:
        frames = snap_frames(round(args.seconds * 24))
    frames_direct = bool(args.frames)

    if args.mode == "r2v":
        wf = json.load(open(JSON_DIR / "H3参考生视频工作流.json"))
        g = Graph(wf, wf["nodes"], wf["links"])
        nm = g.nm
        # 底模/编码器/VAE
        nm[127]["widgets_values"][0] = UNET_REF2VA
        nm[128]["widgets_values"][0] = CLIP_QWEN   # nvfp4 → int8（3090）
        nm[119]["widgets_values"][0] = VAE_VIDEO
        nm[120]["widgets_values"][0] = VAE_AUDIO
        # 提示词（PrimitiveStringMultiline 138 驱动 136）
        nm[138]["widgets_values"][0] = args.prompt
        # 种子 / 采样
        nm[129]["widgets_values"] = [seed, "fixed"]
        nm[123]["widgets_values"] = [args.sampler]
        nm[124]["widgets_values"] = [args.scheduler, args.steps, 1]
        # ref_image_size
        nm[136]["widgets_values"][4] = args.ref_image_size
        if args.shift_video is not None or args.shift_audio is not None:
            insert_sigma_shift(
                g, 127,
                args.shift_video if args.shift_video is not None else 12.0,
                args.shift_audio if args.shift_audio is not None else 3.0)
        # 时长
        if frames_direct:
            idx, inp = g.input_slot(136, "length")
            if inp and inp.get("link") is not None:
                g.cut_link(inp["link"])
            nm[136]["widgets_values"][3] = frames
        else:
            nm[132]["widgets_values"][0] = args.seconds
        # 画布
        patch_size(g, 136, "width", "height", args, widgets_node=nm[136])
        # ---- 参考图：复用模板 137/139，其余新建/删除 ----
        tmpl_img_nodes = [137, 139]
        for i, tag in enumerate(args.ref_image):
            base = up[tag]
            if i < len(tmpl_img_nodes):
                nid = tmpl_img_nodes[i]
                nm[nid]["widgets_values"][0] = base
            else:
                nid = g.add_node("LoadImage", [base, "image"],
                                 [("IMAGE", "IMAGE"), ("MASK", "MASK")],
                                 [nm[136]["pos"][0] - 500, nm[136]["pos"][1] + i * 120])
            idx, _ = g.input_slot(136, f"ref_images.ref_image_{i}")
            g.link(nid, 0, 136, idx, "IMAGE", name=f"ref_images.ref_image_{i}",
                   shape=7, label=f"ref_image_{i}")
        for nid in tmpl_img_nodes[len(args.ref_image):]:
            g.drop_node(nid)
        # ---- 参考视频：LoadVideo → GetVideoComponents → 图像+音轨 ----
        for i, tag in enumerate(args.ref_video):
            lv = g.add_node("LoadVideo", [up[tag]], [("VIDEO", "VIDEO")],
                            [nm[136]["pos"][0] - 700, nm[136]["pos"][1] + 400 + i * 200])
            gvc = g.add_node("GetVideoComponents", [],
                             [("IMAGE", "IMAGE"), ("AUDIO", "AUDIO"),
                              ("fps", "FLOAT"), ("bit_depth", "INT")],
                             [nm[136]["pos"][0] - 380, nm[136]["pos"][1] + 400 + i * 200])
            g.link(lv, 0, gvc, 0, "VIDEO", name="video")
            idx, _ = g.input_slot(136, f"ref_videos.ref_video_{i}")
            g.link(gvc, 0, 136, idx, "IMAGE", name=f"ref_videos.ref_video_{i}",
                   shape=7, label=f"ref_video_{i}")
            idx, _ = g.input_slot(136, f"ref_video_audios.ref_video_audio_{i}")
            g.link(gvc, 1, 136, idx, "AUDIO",
                   name=f"ref_video_audios.ref_video_audio_{i}",
                   shape=7, label=f"ref_video_audio_{i}")
        # ---- 独立参考音频 ----
        for i, tag in enumerate(args.ref_audio):
            la = g.add_node("LoadAudio", [up[tag]], [("AUDIO", "AUDIO")],
                            [nm[136]["pos"][0] - 500, nm[136]["pos"][1] + 800 + i * 120])
            idx, _ = g.input_slot(136, f"ref_audios.ref_audio_{i}")
            g.link(la, 0, 136, idx, "AUDIO", name=f"ref_audios.ref_audio_{i}",
                   shape=7, label=f"ref_audio_{i}")
        mode = "r2v"
    else:
        # fl2va 统一用文生模板；i2v 由脚本动态接首/末帧图覆盖
        wf = json.load(open(JSON_DIR / "H3文生视频工作流.json"))
        g_top = Graph(wf, wf["nodes"], wf["links"])
        sg = wf["definitions"]["subgraphs"][0]
        g_sg = Graph(wf, sg["nodes"], sg["links"])
        patch_common_fl2va(g_top, g_sg, args, seed, frames, frames_direct)
        # 画布（子图实例 105 的 width/height 顶层入槽）
        patch_size(g_top, 105, "width", "height", args,
                   widgets_node=g_top.nm[105])
        # 首帧 / 末帧（动态接线：新建 LoadImage → 105 对应槽）
        for tag, slot_name in (("image", "first_frame"), ("last_image", "last_frame")):
            if getattr(args, tag):
                idx, inp = g_top.input_slot(105, slot_name)
                nid = g_top.add_node("LoadImage", [up[tag], "image"],
                                     [("IMAGE", "IMAGE"), ("MASK", "MASK")],
                                     [g_top.nm[105]["pos"][0] - 500,
                                      g_top.nm[105]["pos"][1]
                                      + (0 if slot_name == "first_frame" else 120)])
                g_top.link(nid, 0, 105, idx, "IMAGE")
        if args.shift_video is not None or args.shift_audio is not None:
            insert_sigma_shift(
                g_sg, 6,
                args.shift_video if args.shift_video is not None else 12.0,
                args.shift_audio if args.shift_audio is not None else 3.0)
        mode = "i2v" if (args.image or args.last_image) else "t2v"

    # 输出前缀（SaveVideo 92，三模板同 id）
    for n in wf["nodes"]:
        if n["type"] == "SaveVideo":
            n["widgets_values"][0] = f"H3_{run_id}"
    return wf, mode, frames


def main():
    ap = argparse.ArgumentParser(
        description="MiniMax H3 全模态视频生成一键跑（T2V/I2V/R2V 自动选择）")
    g_in = ap.add_argument_group("模态输入（fl2va 与 ref2va 互斥）")
    g_in.add_argument("--prompt", required=True, help="提示词（R2V 用 <Picture i>/<Video k>/<Audio j> 引用）")
    g_in.add_argument("--image", help="首帧图（→I2V，fl2va）")
    g_in.add_argument("--last-image", help="末帧图（→I2V，fl2va）")
    g_in.add_argument("--ref-image", action="append", default=[], metavar="IMG",
                      help="参考图（→R2V，≤9，可重复；提示词里按顺序叫 <Picture 1..9>）")
    g_in.add_argument("--ref-video", action="append", default=[], metavar="VID",
                      help="参考视频（→R2V，≤3，可重复，自带音轨即 <Audio>；叫 <Video 1..3>）")
    g_in.add_argument("--ref-audio", action="append", default=[], metavar="AUD",
                      help="独立参考音频（→R2V，≤3，可重复；排在视频音轨之后编号）")
    g_gen = ap.add_argument_group("生成参数")
    g_gen.add_argument("--seconds", type=float, default=5,
                       help="时长秒（默认5；吸附 17k+5 帧网格@24fps；训练范围≈5~15s）")
    g_gen.add_argument("--frames", type=int, default=None,
                       help="直接指定帧数（覆盖 --seconds；吸附 17k+5）")
    g_gen.add_argument("--size", default=None, metavar="WxH",
                       help="画布（32 倍数，如 768x1344；缺省走 --aspect/--megapixels）")
    g_gen.add_argument("--aspect", default="16:9 (Widescreen)", choices=ASPECTS,
                       help="宽高比（默认 16:9）")
    g_gen.add_argument("--megapixels", type=float, default=0.4,
                       help="总像素 MP（默认0.4 快速预览；原生画质≈1.0 → 1344x768）")
    g_gen.add_argument("--steps", type=int, default=20, help="采样步数（默认20）")
    g_gen.add_argument("--sampler", default="res_multistep", help="采样器（默认 res_multistep）")
    g_gen.add_argument("--scheduler", default="simple",
                       help="调度器（默认 simple；参考多的提示词可试 beta/normal）")
    g_gen.add_argument("--seed", type=int, default=None, help="种子（默认随机）")
    g_gen.add_argument("--shift-video", type=float, default=None,
                       help="视频流 flow shift（节点默认12.0；给出时插入 SigmaShift 节点）")
    g_gen.add_argument("--shift-audio", type=float, default=None,
                       help="音频流 flow shift（节点默认3.0；与 --shift-video 可单给）")
    g_gen.add_argument("--ref-image-size", choices=["match", "max"], default="match",
                       help="R2V 参考图缩放：match=跟画布（快）；max=2048 短边（身份更保真，慢数倍）")
    g_run = ap.add_argument_group("运行")
    g_run.add_argument("--out", default=None, help="本地输出路径（默认 ~/Downloads/h3_时间戳.mp4）")
    g_run.add_argument("--no-download", action="store_true", help="只跑不拉回")
    g_run.add_argument("--server", default=SRV)
    g_run.add_argument("--ssh-port", type=int, default=PORT)
    args = ap.parse_args()

    # ---- 模态校验 ----
    fl2va_inputs = bool(args.image or args.last_image)
    ref2va_inputs = bool(args.ref_image or args.ref_video or args.ref_audio)
    assert not (fl2va_inputs and ref2va_inputs), \
        "--image/--last-image（fl2va）与 --ref-*（ref2va）互斥，不能混用"
    assert len(args.ref_image) <= 9, "参考图最多 9 张"
    assert len(args.ref_video) <= 3, "参考视频最多 3 条"
    assert len(args.ref_audio) <= 3, "独立参考音频最多 3 条"
    args.mode = "r2v" if ref2va_inputs else ("i2v" if fl2va_inputs else "t2v")

    run_id = uuid.uuid4().hex[:8]
    seed = args.seed if args.seed is not None else random.randint(0, 2**53 - 1)

    # ---- 上传输入 ----
    orig_refs = (list(args.ref_image), list(args.ref_video), list(args.ref_audio))
    files = []
    if args.image:
        files.append((args.image, "image"))
    if args.last_image:
        files.append((args.last_image, "last_image"))
    files += [(p, f"refimg{i}") for i, p in enumerate(args.ref_image)]
    files += [(p, f"refvid{i}") for i, p in enumerate(args.ref_video)]
    files += [(p, f"refaud{i}") for i, p in enumerate(args.ref_audio)]
    up = upload_inputs(args, run_id, files)
    # 脚本内部用统一 tag 查名
    args.ref_image = [f"refimg{i}" for i in range(len(args.ref_image))]
    args.ref_video = [f"refvid{i}" for i in range(len(args.ref_video))]
    args.ref_audio = [f"refaud{i}" for i in range(len(args.ref_audio))]

    wf, mode, frames = build_workflow(args, run_id, seed, up)
    dur = frames / 24
    shift_info = ""
    if args.shift_video is not None or args.shift_audio is not None:
        sv = args.shift_video if args.shift_video is not None else 12.0
        sa = args.shift_audio if args.shift_audio is not None else 3.0
        shift_info = f" shift=v{sv}/a{sa}"
    print(f"run_id={run_id} mode={mode} seed={seed} 帧数={frames}@24fps（{dur:.2f}s）"
          f" steps={args.steps} {args.sampler}/{args.scheduler}{shift_info}", flush=True)
    if frames > 362:
        print(f"!! {frames} 帧超出训练范围（124~362），画质自负", file=sys.stderr)
    if mode == "r2v":
        o_img, o_vid, o_aud = orig_refs
        nv = len(o_vid)
        tags = ([f"<Picture {i+1}>={Path(p).name}" for i, p in enumerate(o_img)]
                + [f"<Video {i+1}>={Path(p).name}" for i, p in enumerate(o_vid)]
                + [f"<Audio {i+1}>=视频{i+1}音轨" for i in range(nv)]
                + [f"<Audio {nv+i+1}>={Path(p).name}" for i, p in enumerate(o_aud)])
        print("参考标签映射（提示词里用同名标签）:\n  " + "\n  ".join(tags), flush=True)

    tmp = Path("/tmp/run_h3_workflow.json")
    json.dump(wf, open(tmp, "w"), ensure_ascii=False)
    scp_to(args.server, args.ssh_port, str(tmp), "~/run_h3_workflow.json")

    # ---- 提交执行 ----
    ssh(args.server, args.ssh_port,
        "export PATH=$HOME/.local/bin:$PATH; "
        "setsid nohup comfy run --workflow ~/run_h3_workflow.json "
        "--host 127.0.0.1 --port 8188 --wait --verbose --timeout 10800 "
        "> /tmp/run_h3.log 2>&1 </dev/null & disown; echo submitted")
    print("已提交，轮询产物...", flush=True)

    # ---- 轮询产物并拉回 ----
    # 完成信号 = comfy run 进程退出（文件出现≠写完，moov 最后才落盘）。
    t0 = time.time()
    out_glob = f"{SVR_OUTPUT}/H3_{run_id}*"
    while True:
        time.sleep(20)
        done = ssh(args.server, args.ssh_port,
                   "grep -qE '\"ok\": false|Traceback' /tmp/run_h3.log 2>/dev/null && echo FAIL || echo OK",
                   capture=True, retries=6).stdout.strip()
        if done == "FAIL":
            err = ssh(args.server, args.ssh_port,
                      "tail -15 /tmp/run_h3.log", capture=True).stdout
            print("!! 执行失败\n", err, file=sys.stderr)
            sys.exit(3)
        running = ssh(args.server, args.ssh_port,
                      "pgrep -f 'comfy r[u]n --workflow' >/dev/null && echo y || echo n",
                      capture=True, retries=6).stdout.strip()
        if running != "y":
            break
        alive = ssh(args.server, args.ssh_port,
                    "pgrep -f '[m]ain.py' >/dev/null && echo y || echo n",
                    capture=True, retries=6).stdout.strip()
        if alive != "y":
            err = ssh(args.server, args.ssh_port,
                      "tail -8 /tmp/run_h3.log", capture=True).stdout
            print("!! ComfyUI 进程消失\n", err, file=sys.stderr)
            sys.exit(2)
        if time.time() - t0 > 10800:
            print("!! 轮询超时（3h）", file=sys.stderr)
            sys.exit(5)
        print(f"  ... {int(time.time()-t0)}s", flush=True)

    cur = ssh(args.server, args.ssh_port,
              f"ls -t {out_glob} 2>/dev/null | head -1",
              capture=True, retries=6).stdout.strip()
    if not cur:
        err = ssh(args.server, args.ssh_port,
                  "tail -15 /tmp/run_h3.log", capture=True).stdout
        print("!! 流程结束但找不到产物\n", err, file=sys.stderr)
        sys.exit(4)
    print("产物就绪:", cur, flush=True)

    if not args.no_download:
        out = args.out or str(Path.home() / "Downloads" /
                              f"h3_{mode}_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
        tmp_remote = f"/tmp/h3_out_{run_id}{os.path.splitext(cur)[1] or '.mp4'}"
        for attempt in (1, 2):
            ssh(args.server, args.ssh_port, f"cp '{cur}' {tmp_remote}")
            sh(["scp", *SSH_OPTS, "-P", str(args.ssh_port),
                f"{args.server}:{tmp_remote}", out])
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-count_frames",
                 "-show_entries", "stream=codec_type,codec_name,width,height,"
                 "r_frame_rate,nb_read_frames,channels",
                 "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1", out],
                capture_output=True, text=True)
            ok_v = "codec_type=video" in probe.stdout and "nb_read_frames=0" not in probe.stdout
            if probe.returncode == 0 and ok_v:
                break
            print(f"!! 产物校验失败（第{attempt}次），5s 后重拉\n"
                  f"{probe.stderr.strip()[:300]}", flush=True)
            time.sleep(5)
        else:
            print("!! 产物仍不可读，服务器原件保留在 " + cur, file=sys.stderr)
            sys.exit(4)
        if "codec_type=audio" not in probe.stdout:
            print("!! 警告：产物没有音轨（H3 应原生出声）", file=sys.stderr)
        ssh(args.server, args.ssh_port,
            f"rm -f {tmp_remote} {SVR_INPUT}/h3_{run_id}_*")
        print(probe.stdout.strip())
        print("DONE ->", out)
    else:
        print("DONE (no-download)")


if __name__ == "__main__":
    main()
