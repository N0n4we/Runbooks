#!/usr/bin/env python3
"""
run-h3.py — MiniMax H3 全模态视频生成一键跑（多目标机，见 HOSTS / --host）

目标机档位（--host，缺省 kk14590）：
  5090  PER3cU@wp08:21054  RTX 5090 32G sm_120  → int8 convrot（triton 加速）
  4090  kIYRa5@wp08:25304  RTX 4090 24G sm_89   → int4/mixed convrot（eager）
  kk14590 KKujSt@wp08:14590 RTX 4090 24G sm_89  → INT8 ConvRot + NVFP4 encoder + Larry

底图：四个新工作流（Qwen3VL 版，均含 VideoHelperSuite / KJNodes / Easy-Use /
Upscaler-Tensorrt / Rife-Tensorrt / AILab QwenVL 等第三方节点）：
  MiniMaxH3-T2VA-Qwen3VL.json   纯文本 → 视频+音频
  MiniMaxH3-I2VA-Qwen3VL.json   首帧图 → 视频+音频（根图 LoadImage #114）
  MiniMaxH3-FL2VA-Qwen3VL.json  首尾帧 → 视频+音频（LoadImage #114 首 / #147 尾）
  MiniMaxH3-R2VA-Qwen3VL.json   参考图 → 视频+音频（MiniMaxH3ReferenceToVideo）
四者推理链均封装在同一子图（实例节点 #105 / definitions.subgraphs[0]）。

模态自动选择（互斥）：
  --prompt                                       → MiniMaxH3-T2VA
  --prompt + --image                             → MiniMaxH3-I2VA
  --prompt + --image + --last-image              → MiniMaxH3-FL2VA 可以不传入--image
  --prompt + --ref-image/--ref-video/--ref-audio → MiniMaxH3-R2VA 当前仅支持--ref-image，最多2张

  python3 run-h3.py --prompt "……"
  python3 run-h3.py --prompt "……" --image first.png --last-image last.png
  python3 run-h3.py --prompt "让 <Picture 1> 的角色走进 <Picture 2> 的场景……" \
      --ref-image role.png --ref-image scene.png

参考标签约定（R2V 提示词里引用，1 起始）：
  <Picture i>  参考图，按 --ref-image 顺序（当前模板上限 2，见下）

!! R2V 参考能力 = 2 张参考图（对齐工作流，非缺口）：
   核心节点 MiniMaxH3ReferenceToVideo(#149) 自身有 ref_image_0/1/2、ref_video_0、
   ref_video_audio_0、ref_audio_0 六个参考槽，但 MiniMaxH3-R2VA-Qwen3VL.json 只接了
   前两个：子图边界仅暴露 ref_images.ref_image_0 / ref_image_1，其余四槽 link=None，
   整个工作流（子图 + 根图）没有任何 LoadVideo / LoadAudio 节点，根图恰好两个
   LoadImage(#114/#152)。**脚本按此对齐**：--ref-image 上限 2 张，--ref-video /
   --ref-audio 传入即报错退出（此前会被静默上传后丢弃，产出忽略参考视频/音频的结果
   且无任何报错——那是最坏的失败方式）。要超出这个能力得改模板本身，不在本脚本职责内。

要点：
  - 帧数网格 17k+5 @24fps（5s=124 帧）；训练范围 124~362 帧（≈5~15s）
  - 画布 32 倍数；原生 768 短边、面积上限 768×1344（≈0.4MP 模板默认快速预览）
  - BasicGuider=cfg1 无负面词；采样 res_multistep + simple 20 步（模板默认）
  - R2V 参考能力：2 张参考图。这是 MiniMaxH3-R2VA-Qwen3VL.json 的设计能力（子图边界只
    暴露 ref_image_0/1，核心 #149 的 ref_image_2 与 ref_video/ref_audio 槽未连线，
    工作流里也没有 LoadVideo/LoadAudio 节点），不是待修复的缺口。
  - --shift-video/--shift-audio：插入 MiniMaxH3SigmaShift 节点（接管 UNET 全部
    MODEL 出边，调度器与引导器用同一份 patched model）调视频/音频流 flow shift
  - --lora NAME[:strength]：UNETLoader 后插 LoraLoaderModelOnly 接管 UNET 全部 MODEL
    出边（与 shift 同技术；两者叠加时链路为 UNET→LoRA→SigmaShift→引导器/调度器）
"""
import argparse, json, os, random, subprocess, sys, time, uuid
from pathlib import Path

# 目标机档位。两台机器的量化选型不同，且**不是偏好问题而是内核可用性问题**：
# comfy-kitchen 的 cuda 后端被 ComfyUI 以 "torch cuda < 13" 为由禁用（comfy/quant_ops.py），
# 于是唯一的加速后端是 triton，而 triton 后端只有 int8_linear / w4a8_int8_linear /
# fp8，**没有 convrot_w4a4_linear、也没有 nvfp4 的 linear** → int4 与 NVFP4 都只能走
# eager 反量化。5090(sm_120) 上实测同样如此：NVFP4 虽是 Blackwell 原生格式，在 cu128
# 上仍落到 eager，故 32G 机的最优解是 int8 而非 NVFP4。
HOSTS = {
    "5090": {"srv": "PER3cU@wp08.unicorn.org.cn", "port": 21054,
             "dit": "int8",  "clip": "int8",  "vram": 32},
    "4090": {"srv": "kIYRa5@wp08.unicorn.org.cn", "port": 25304,
             "dit": "int4",  "clip": "int4",  "vram": 24},
    # 用户指定的现役部署目标：24G 4090。NVFP4-AWQ encoder 约15.7G，
    # pruned INT8 ConvRot DiT 约21G，分别整载到显存可行；同一任务中由 ComfyUI 串行换载。
    "kk14590": {"srv": "KKujSt@wp08.unicorn.org.cn", "port": 14590,
                "dit": "int8", "clip": "nvfp4", "vram": 24},
}
DEFAULT_HOST = "kk14590"
SRV, PORT = HOSTS[DEFAULT_HOST]["srv"], HOSTS[DEFAULT_HOST]["port"]
SSH_OPTS = ["-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15"]  # TOFU：首连收指纹并记录；绝不用 no
JSON_DIR = Path(__file__).parent
SVR_INPUT = "~/comfy/ComfyUI/input"
SVR_OUTPUT = "~/comfy/ComfyUI/output"

# 模态 → 新工作流模板（task 2）。四者结构一致：推理链封装在同一子图
# （实例节点 #105，子图定义 definitions.subgraphs[0]），根图仅 13–15 节点
# （子图实例 + Upscaler/Rife TensorRT 后处理链 + VHS_VideoCombine + SaveVideo
#  + ResolutionSelector + easy showAnything + 若干 MarkdownNote）。
TEMPLATES = {
    "t2v":   "MiniMaxH3-T2VA-Qwen3VL.json",   # 纯文本 → 视频+音频
    "i2v":   "MiniMaxH3-I2VA-Qwen3VL.json",   # 首帧图 → 视频+音频（根图 LoadImage #114）
    "fl2va": "MiniMaxH3-FL2VA-Qwen3VL.json",  # 首尾帧 → 视频+音频（LoadImage #114 首 / #147 尾）
    "r2v":   "MiniMaxH3-R2VA-Qwen3VL.json",   # 参考图 → 视频+音频（MiniMaxH3ReferenceToVideo）
}


def select_template(mode):
    """按模态返回模板绝对路径；文件缺失即时报错。"""
    p = JSON_DIR / TEMPLATES[mode]
    if not p.exists():
        sys.exit(f"!! 模板缺失：{p}")
    return p

# DiT 量化档位。注意**两个来源的文件命名不同**，不可混用：
#   Comfy-Org/MiniMax-H3（官方，ModelScope）  全小写 minimax_h3_*   → int8 / fp8
#   Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot（第三方）大写 MiniMax_H3_* → int4 / mixed
# 加速性（实测见 HOSTS 注释）：int8 与 fp8 走 triton 加速；int4/mixed 的 w4a4 段落到
# eager。故 32G 机用 int8，24G 机因装不下 int8 编码器才退 int4/mixed。
DIT_VARIANTS = {
    # 官方（全小写）——5090 机已下载
    "int8":  ("minimax_h3_fl2va_pruned_int8_convrot.safetensors",
              "minimax_h3_ref2va_pruned_int8_convrot.safetensors"),      # 19.53G each
    "fp8":   ("minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
              "minimax_h3_ref2va_pruned_fp8_scaled.safetensors"),        # 19.52G each
    # 第三方（大写）——4090 机已下载
    "int4":  ("MiniMax_H3_FL2VA_pruned_int4_convrot.safetensors",
              "MiniMax_H3_Ref2VA_pruned_int4_convrot.safetensors"),
    "mixed": ("MiniMax_H3_FL2VA_pruned_mixed_int4_int8_convrot.safetensors",
              "MiniMax_H3_Ref2VA_pruned_mixed_int4_int8_convrot.safetensors"),
}
# 文本编码器档位（Qwen3VL-32B，Comfy-Org 官方仓库）。这是选型的真正瓶颈：
#   int8 25.28G → 只有 32G 卡能整块驻留；24G 卡上会被迫 offload 到 CPU（极慢）
#   int4 13.93G → 24G 卡的唯一可整块驻留档
CLIP_VARIANTS = {
    "int8": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",   # 25.28G
    "int4": "qwen3vl_32b_minimax_h3_int4_convrot.safetensors",   # 13.93G
    "nvfp4": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",     # 15.7G
}
# 运行期由 --host / --dit / --clip 覆盖（见 main()）
UNET_FL2VA, UNET_REF2VA = DIT_VARIANTS[HOSTS[DEFAULT_HOST]["dit"]]
CLIP_QWEN = CLIP_VARIANTS[HOSTS[DEFAULT_HOST]["clip"]]
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


def pull_output(server, port, glob_pattern, dest, run_id, label, retries=2):
    """找 glob 下最新产物 → scp 拉回 → ffprobe 校验（moov 后写，失败重拉）→ 清理 /tmp 副本。
    返回 (ok, out_path, probe)。ok=False 表示没找到或不可读（ComfyUI 未写到含 moov 的完整文件）。
    成功路径与失败抢救共用，保证两条路径行为一致。"""
    cur = ssh(server, port,
              f"ls -t {glob_pattern} 2>/dev/null | head -1",
              capture=True, retries=6).stdout.strip()
    if not cur:
        return False, None, None
    tmp = f"/tmp/pull_{run_id}_{label}{os.path.splitext(cur)[1] or '.mp4'}"
    probe, ok = None, False
    for attempt in (1, retries):
        ssh(server, port, f"cp '{cur}' {tmp}")
        sh(["scp", *SSH_OPTS, "-P", str(port), f"{server}:{tmp}", dest])
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames",
             "-show_entries", "stream=codec_type,codec_name,width,height,"
             "r_frame_rate,nb_read_frames,channels",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1", dest],
            capture_output=True, text=True)
        ok = (probe.returncode == 0 and "codec_type=video" in probe.stdout
              and "nb_read_frames=0" not in probe.stdout)
        if ok:
            break
        print(f"!! {label}产物校验失败（第{attempt}次），5s 后重拉\n"
              f"{probe.stderr.strip()[:300]}", flush=True)
        time.sleep(5)
    ssh(server, port, f"rm -f {tmp}")
    return ok, dest, probe


def salvage_core_output(server, port, out_glob, run_id, mode, out=None):
    """run 失败时尽力抢救核心(未超分)产物：SaveVideo#92 通常已先于超分链写好 H3_<id>*。
    best-effort：不改变失败退出码，--no-download 时调用方保证不调用。"""
    dest = out or str(Path.home() / "Downloads" /
                      f"h3_{mode}_{time.strftime('%Y%m%d_%H%M%S')}_core.mp4")
    ok, path, probe = pull_output(server, port, out_glob, dest, run_id, "salvage")
    if not ok:
        print("!! 抢救核心产物失败（原始视频未写出或不可读）", file=sys.stderr)
        return
    if "codec_type=audio" not in probe.stdout:
        print("!! 警告：抢救的产物没有音轨", file=sys.stderr)
    print(probe.stdout.strip())
    print("!! 本次 run 失败，已抢救原始(未超分)视频 ->", path, file=sys.stderr)


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


def insert_lora(g, unet_id, lora_name, strength):
    """task 8：在 UNETLoader 后插 LoraLoaderModelOnly，接管其全部 MODEL 出边（与
    SigmaShift 同技术）。patched model 单份下传——BasicGuider 与 BasicScheduler 共用同
    一份 LoRA 后的 model，采样与调度口径才一致。仅接管 MODEL 边，CLIP/VAE 不受影响。

    与 SigmaShift 叠加时本函数在其之后调用：此刻 UNET 的唯一 MODEL 出边指向 SigmaShift，
    故 LoRA 落在 UNET 与 SigmaShift 之间，链路为 UNET → LoRA → SigmaShift → {引导器/调度器}
    （先按 LoRA 改权重，再按 SigmaShift 调 sigma）；无 SigmaShift 时即 UNET → LoRA → 两下游。"""
    pos = g.nm[unet_id]["pos"]
    lid = g.add_node("LoraLoaderModelOnly", [lora_name, strength],
                     [("MODEL", "MODEL")], [pos[0] + 180, pos[1] - 60],
                     title=f"LoRA: {lora_name}")
    targets = [(g._l(l, "id"), g._l(l, "target_id"), g._l(l, "target_slot"))
               for l in list(g.links)
               if g._l(l, "origin_id") == unet_id and g._l(l, "type") == "MODEL"]
    for old_lid, tgt, tslot in targets:
        g.cut_link(old_lid)
    g.link(unet_id, 0, lid, 0, "MODEL", name="model")
    for _, tgt, tslot in targets:
        g.link(lid, 0, tgt, tslot, "MODEL")
    return lid


def insert_larry_lora(g, unet_id, sampler_id, lora_name, strength):
    """插入 Larry H3 Turbo 的专用 LoRA + sampler 节点。

    Larry 的 LoRA 不是只把权重接到普通 LoraLoaderModelOnly 就结束：它还提供
    MiniMaxH3TurboSampler，用于 4--8 步时的 H3 视频/音频双时钟采样。模型链仍
    按普通 MODEL patch 方式接管 UNET 的全部出边，因此和 SigmaShift 可叠加：
    UNET → Larry → SigmaShift → {BasicScheduler/BasicGuider}。
    """
    pos = g.nm[unet_id]["pos"]
    lid = g.add_node("MiniMaxH3TurboLoRA", [lora_name, strength, False],
                     [("MODEL", "MODEL")], [pos[0] + 180, pos[1] - 60],
                     title=f"Larry H3 Turbo: {lora_name}")
    targets = [(g._l(l, "id"), g._l(l, "target_id"), g._l(l, "target_slot"))
               for l in list(g.links)
               if g._l(l, "origin_id") == unet_id and g._l(l, "type") == "MODEL"]
    for old_lid, _tgt, _tslot in targets:
        g.cut_link(old_lid)
    g.link(unet_id, 0, lid, 0, "MODEL", name="model")
    for _, tgt, tslot in targets:
        g.link(lid, 0, tgt, tslot, "MODEL")

    # The custom sampler replaces KSamplerSelect's SAMPLER output at the
    # SamplerCustomAdvanced node; its scheduler/sigmas remain unchanged.
    sid = g.add_node("MiniMaxH3TurboSampler", [], [("SAMPLER", "SAMPLER")],
                     [pos[0] + 520, pos[1] + 180], title="Larry H3 Turbo Sampler")
    _idx, sinp = g.input_slot(sampler_id, "sampler")
    if sinp is not None and sinp.get("link") is not None:
        g.cut_link(sinp["link"])
    g.link(sid, 0, sampler_id, _idx, "SAMPLER")
    return lid, sid


def parse_lora(spec):
    """--lora NAME[:strength] → (name, strength)。strength 缺省 1.0。
    rsplit(':',1) 容忍文件名/路径里出现的其它冒号，只把末段当强度。"""
    name, sep, s = spec.rpartition(":")
    if not sep:
        return spec, 1.0
    try:
        return name, float(s)
    except ValueError:
        # 末段不是数字（例如 Windows 盘符或本就无强度），整串当文件名
        return spec, 1.0


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


def apply_params(g_top, g_sg, args, seed, frames, frames_direct, mode):
    """task 5：把 尺寸/时长/帧数/步数/采样器/调度器/种子/shift/ref_image_size 写进对应节点。

    双写原则：子图实例 #105 的 promoted widgets 是权威值（执行期经 link 喂进子图内部、
    覆盖内部同名 widget），内部节点再按需双写一份兜底（防前端/后端转换器口径不一，也便于
    直接读内部图校验）。实例 promoted widget 顺序（四模板一致，first/last/ref_image 走 link
    不占 widget）见文件顶部 PROMOTED 注释；本函数用到的：[3]duration_sec [4]noise_seed
    [12]steps。主推理节点：t2v/i2v/fl2va → #104 MiniMaxH3ImageToVideo；
    r2v → #149 MiniMaxH3ReferenceToVideo。

    刻意不碰（属其它任务，勿越界）：权重名(task 3) / model_name·preset_prompt·keep_last
    (task 4) / prompt 与 QwenVL 旁路(task 7)。--lora(task 8) 在本函数 shift 之后插入
    LoraLoaderModelOnly。尺寸(width/height)由外层的 patch_size 在顶层图重接线，本函数
    只把内部 core 的 w/h widget 同步过去。
    """
    inst = g_top.nm[105]                      # 子图实例（promoted widgets 权威）
    sn = g_sg.nm                              # 子图内部节点
    core_id = 149 if mode == "r2v" else 104   # r2v 用 ReferenceToVideo，其余用 ImageToVideo

    # --- 种子：promoted widget[4] 权威（经 link 喂 RandomNoise）；RandomNoise #15 双写并
    #     锁 fixed（模板默认 control_after_generate=randomize 会每跑漂移，锁 fixed 才复现）---
    inst["widgets_values"][4] = seed
    sn[15]["widgets_values"] = [seed, "fixed"]

    # --- 步数：promoted widget[12] 权威（经 link 喂 BasicScheduler.steps）；#9 的 steps 位双写。
    #     （task4 的 passthrough_new_inputs 亦写同值，此处幂等重申，保 task5 自洽）---
    inst["widgets_values"][12] = args.steps
    sn[9]["widgets_values"][1] = args.steps

    # --- 采样器 / 调度器：子图内部专属，无 promoted，直写（按索引写，不整块换以保 denoise）---
    sn[17]["widgets_values"][0] = args.sampler       # KSamplerSelect
    sn[9]["widgets_values"][0] = args.scheduler      # BasicScheduler [scheduler, steps, denoise]

    # --- 尺寸内部双写：显式 --size 时由 patch_size 在顶层重接线并双写实例 widget[1/2]，
    #     此处把内部 core 的 width/height widget 同步为实例值（缺省走 ResolutionSelector
    #     运行期定尺寸，故仅在 --size 显式时同步，避免写入过期默认值误导）---
    if args.size:
        sn[core_id]["widgets_values"][1] = inst["widgets_values"][1]
        sn[core_id]["widgets_values"][2] = inst["widgets_values"][2]

    # --- 时长 / 帧数 ---
    if frames_direct:
        # --frames 直写 length：剪断 ComfyMathExpression→core 的秒→帧内部边，写 core.length widget
        idx, inp = g_sg.input_slot(core_id, "length")
        if inp and inp.get("link") is not None:
            g_sg.cut_link(inp["link"])
        sn[core_id]["widgets_values"][3] = frames
    else:
        # 秒链：promoted duration widget[3] 权威；PrimitiveFloat #111 双写
        inst["widgets_values"][3] = args.seconds
        sn[111]["widgets_values"][0] = args.seconds

    # --- ref_image_size：仅 R2V，MiniMaxH3ReferenceToVideo #149 内部 widget[4]（match/max）---
    if mode == "r2v":
        sn[149]["widgets_values"][4] = args.ref_image_size

    # --- flow shift：给出 --shift-video/--shift-audio 任一即插入 MiniMaxH3SigmaShift，
    #     接管 UNET #6 全部 MODEL 出边（调度器与引导器共用同一 patched model）---
    if args.shift_video is not None or args.shift_audio is not None:
        sv = args.shift_video if args.shift_video is not None else 12.0
        sa = args.shift_audio if args.shift_audio is not None else 3.0
        insert_sigma_shift(g_sg, 6, sv, sa)

    # --- LoRA（task 8）：Larry Turbo 自动使用专用节点；其它文件保持原生节点兼容。---
    if args.lora:
        lname, lstrength = parse_lora(args.lora)
        use_larry = (args.lora_backend == "larry" or
                     (args.lora_backend == "auto" and
                      any(x in lname.lower() for x in ("larry", "turbo"))))
        if use_larry:
            insert_larry_lora(g_sg, 6, 14, lname, lstrength)
        else:
            insert_lora(g_sg, 6, lname, lstrength)


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


def rewrite_weights(g_top, g_sg, mode):
    """task 3：把模板写死的 NVFP4 权重名改写成当前主机档位可用的量化权重。

    模板引用的 NVFP4（10Eros…FL2VA-NVFP4 / MiniMax-H3_Ref2VA-NVFP4 / qwen3vl…nvfp4）
    在 cu128 上**两台机器都没有加速核**：NVFP4 的 linear 只在 comfy-kitchen 的 cuda
    后端，而该后端被 ComfyUI 以 "torch cuda < 13" 禁用；triton 后端无 nvfp4 linear。
    5090(sm_120) 也不例外——原生格式≠有内核。改写规则：
      - DiT 按模态选：r2v → UNET_REF2VA，其余(t2v/i2v/fl2va) → UNET_FL2VA
        （档位由 --host/--dit 落到全局 UNET_FL2VA/UNET_REF2VA，见 main）。
      - 编码器 nvfp4 → CLIP_QWEN（由 --host/--clip 决定 int8/int4 档），
        CLIPLoader 的 type 第二个 widget 保持 'minimax' 不动。
      - 视频/音频 VAE 名模板本就正确，一并双写保证一致。
    双写实例 #105 promoted widgets 与子图内部 loader 节点（防前端/后端转换器口径不一）。
    R2VA 分叉修正：实例 promoted[5] 已是 Ref2VA，但子图内 UNETLoader #6 残留 FL2VA
    默认值——两处统一写成正确的 Ref2VA 权重。
    """
    unet = UNET_REF2VA if mode == "r2v" else UNET_FL2VA
    inst = g_top.nm[105]                       # 子图实例（promoted widgets）
    inst["widgets_values"][5] = unet           # unet_name
    inst["widgets_values"][6] = CLIP_QWEN      # clip_name（nvfp4 → int4_convrot）
    inst["widgets_values"][7] = VAE_VIDEO      # vae_name（video）
    inst["widgets_values"][8] = VAE_AUDIO      # vae_name_1（audio）
    sn = g_sg.nm                               # 子图内部 loader 节点
    sn[6]["widgets_values"][0]  = unet         # UNETLoader
    sn[13]["widgets_values"][0] = CLIP_QWEN    # CLIPLoader（[1]=type 保持 'minimax'）
    sn[11]["widgets_values"][0] = VAE_VIDEO    # VAELoader（video）
    sn[24]["widgets_values"][0] = VAE_AUDIO    # VAELoader（audio）
    # loader 的 properties.models[*] 是前端/Manager 的下载提示元数据（headless comfy
    # run 不据此加载，但同步过来免得图里残留 NVFP4 文件名，且保持自洽）：name 改新权重名；
    # 同时删掉模板残留的 HF url——它既是失效链接（HF 不可达，权重实际取自 ModelScope），
    # 又把旧 NVFP4 basename 留在图里、与刚改好的 name 自相矛盾，删掉即消除残留、恢复自洽。
    for nid, name in ((6, unet), (13, CLIP_QWEN), (11, VAE_VIDEO), (24, VAE_AUDIO)):
        for m in sn[nid].get("properties", {}).get("models", []):
            m["name"] = name
            m.pop("url", None)
    return unet


# 子图实例 #105 promoted widget 顺序（四模板一致，first/last/ref_image 走 link 不占 widget）：
#   [0]prompt [1]width [2]height [3]duration_sec [4]noise_seed [5]unet_name
#   [6]clip_name [7]video_vae [8]audio_vae [9]model_name [10]preset_prompt
#   [11]keep_last_prompt [12]steps
# 后四项是四个新模板独有的子图新增输入（旧模板没有），task 4 在此透传。
PROMOTED = {"model_name": 9, "preset_prompt": 10, "keep_last_prompt": 11, "steps": 12}
# 内部增强器节点 id 与 preset_prompt 内部 widget 索引（随节点类型漂移）。model_name
# 恒在增强器 widget[0]；steps 在 BasicScheduler(9) widget[1]；keep_last_prompt 的内部
# 副本索引随节点类型漂移且与 promoted 背离（实测 t2v 内部=True / promoted=False，执行
# 走 link 用 promoted），故只写 promoted 权威源。
ENHANCER = {   # mode -> (增强器节点 id, preset_prompt 内部 widget 索引)
    "t2v":   (151, 6),   # AILab_QwenVL_PromptEnhancer
    "i2v":   (150, 5),   # AILab_QwenVL_Advanced
    "fl2va": (151, 5),   # AILab_QwenVL_Advanced
    "r2v":   (154, 5),   # AILab_QwenVL_Advanced
}
# 用户 prompt 在增强器节点内的 widget 索引（节点类型不同，名字与位置都不同）：
#   t2v   AILab_QwenVL_PromptEnhancer → prompt_text   = widget[5]
#   其余  AILab_QwenVL_Advanced       → custom_prompt = widget[6]
# 增强器缺省被旁路，执行期真正生效的是子图实例 #105 的 promoted widget[0]（见 inject_prompt）；
# 这里仍要写，是为了 --enhance 恢复增强链时 prompt 不丢。
PROMPT_WIDGET = {"t2v": 5, "i2v": 6, "fl2va": 6, "r2v": 6}
# task 7 后处理链（根图，四模板 id 一致，见 NODE_INDEX）：TensorRT 上采样 + RIFE 插帧
# + VHS 二次封装。核心链（子图 VIDEO 输出 → SaveVideo #92，link 194）不经此链，故整条
# 可整体旁路而不影响 h264+aac 出片。
POST_NODES = [
    128,  # LoadUpscalerTensorrtModel（Upscaler-Tensorrt，需 .engine）
    127,  # UpscalerTensorrt
    146,  # AutoLoadRifeTensorrtModel（Rife-Tensorrt，需 .engine）
    145,  # AutoRifeTensorrt
    126,  # VHS_VideoCombine（依赖上面两条链的 IMAGE 输入）
]
# 后处理链按「能不能用」拆成两段（--upscale 只启用超分侧，见 apply_degradation）：
#   超分侧：节点名与已装 ComfyUI-Upscaler-Tensorrt 一致，装上 tensorrt 即可用
#   插帧侧：模板引用 AutoRifeTensorrt / AutoLoadRifeTensorrtModel，而已装包只提供
#           RifeTensorrt / LoadRifeTensorrtModel → 名字不匹配，激活即被服务端拒
UPSCALE_LOAD, UPSCALE_NODE = 128, 127
RIFE_NODES = [146, 145]
VHS_NODE = 126
# 超分引擎的 shape 范围（来自节点包 load_upscaler_config.json）：build 出来的 engine 只在
# 此区间内有效，超了会在推理期报错，故 main() 里按此拦截。
UPSCALE_DIM_MIN, UPSCALE_DIM_MAX = 256, 1280
MODE_ACTIVE, MODE_BYPASS = 0, 4  # ComfyUI 节点 mode：0=正常，4=bypass（按类型透传出边）

# 根图 LoadImage 节点 id（按模态）→ 上传 tag。模板 widget 里写死的是示例图名，需改写成
# upload_inputs 落定的 run_id 前缀远端 basename。LoadImage→子图 first_frame/last_frame 的
# link 模板已连好，仅 widget[0] 需改写（结构不动）。r2v 模板仅两个 LoadImage 槽（#114/#152）。
LOADIMAGE_NODES = {
    "t2v":   {},
    "i2v":   {114: "image"},
    "fl2va": {114: "image", 147: "last_image"},
    "r2v":   {114: "refimg0", 152: "refimg1"},
}


def inject_load_images(g_top, up, mode):
    """把已上传输入的远端 basename 写进模板 LoadImage 节点 widget[0]（模板原为示例图名）。

    **未提供对应输入的 LoadImage 必须旁路，不能放任不管**：模板 widget[0] 里写死的是
    作者机器上的示例图名（如 r2v #152 的 's02_ep06_unexpectedcall_371_v01.jpg'），
    该文件在目标机 input/ 下不存在 → 服务端 combo 校验直接判 unknown_enum_value、
    整个 prompt 被拒（实测 r2v 只给 1 张 --ref-image 时必发）。节点 #149 的
    ref_images/ref_videos/ref_video_audios/ref_audios 均为 optional，故旁路安全：
    被旁路的 LoadImage 不执行，对应 ref_image_N 槽保持未连接。
    不能改用「塞一张已上传的图凑数」——那会凭空多出一个参考图、改变生成结果。

    dry-run 时 up 为空 → 全部保留模板默认且不旁路（仅供结构自检，不提交服务端）。
    返回 (写入映射, 被旁路的节点 id 列表)。
    """
    written, bypassed = {}, []
    if not up:                                   # dry-run：不改写、不旁路
        return written, bypassed
    for nid, tag in LOADIMAGE_NODES.get(mode, {}).items():
        if nid not in g_top.nm:
            continue
        if tag in up:
            g_top.nm[nid]["widgets_values"][0] = up[tag]
            written[nid] = up[tag]
        else:
            _set_mode(g_top, nid, MODE_BYPASS)
            bypassed.append(nid)
    return written, bypassed


def rewire_core_prompt_to_raw(g_sg, mode, core_id):
    """增强器旁路时，把核心节点 prompt 输入直接接到子图 raw prompt 边界(-10, prompt 槽)。

    起因：comfy-cli 对 mode=4 旁路节点按**槽位序号**透传出边，而 AILab_QwenVL_Advanced 的
    STRING 输出在 slot0、其唯一 STRING 输入(custom_prompt)却在 slot4，slot0 输入是 IMAGE
    → 核心节点 prompt 收到 IMAGE（类型不符，服务端校验报 edge_type_mismatch / 执行必崩）。
    故显式重接：核心 prompt ← 子图 prompt 边界（原始 prompt STRING），绕开被旁路增强器。
    保留 enhancer→-20(showAnything 预览, * 型) 的原边不动（* 接受任意类型，无害）。"""
    sg = g_sg.wf["definitions"]["subgraphs"][0]
    pin = next((i for i, x in enumerate(sg["inputs"]) if x.get("name") == "prompt"), 0)
    cidx, cinp = g_sg.input_slot(core_id, "prompt")
    if cinp is None:
        return None
    old = cinp.get("link")
    if old is not None:
        g_sg.cut_link(old)                       # 摘旧边（增强器输出 links 记账 + core 输入清空）
    lid = g_sg.lid()
    sg["links"].append({"id": lid, "origin_id": -10, "origin_slot": pin,
                        "target_id": core_id, "target_slot": cidx, "type": "STRING"})
    cinp["link"] = lid
    sg["inputs"][pin].setdefault("linkIds", []).append(lid)   # 维护边界 linkIds 记账
    return lid


def passthrough_new_inputs(g_top, g_sg, args, mode):
    """透传四个新模板独有的子图新增输入 model_name/preset_prompt/keep_last_prompt/steps。

    权威落点是子图实例 #105 的 promoted widgets（子图边界，被连内部节点执行期读 link
    值即读它）。按本仓“双写”约定，再把能无歧义定位的内部副本一并写上：model_name→
    增强器 widget[0]、preset_prompt→增强器按节点类型的 widget、steps→BasicScheduler(9)
    widget[1]。缺省沿用模板原值，CLI 给出即覆盖。返回落定的 4 值供打印/校验。"""
    inst = g_top.nm[105]
    wv = inst["widgets_values"]
    model_name = args.model_name if args.model_name is not None else wv[PROMOTED["model_name"]]
    preset     = args.preset_prompt if args.preset_prompt is not None else wv[PROMOTED["preset_prompt"]]
    keep_last  = args.keep_last_prompt if args.keep_last_prompt is not None else wv[PROMOTED["keep_last_prompt"]]
    steps      = args.steps
    # 1) promoted widgets（权威）
    wv[PROMOTED["model_name"]]       = model_name
    wv[PROMOTED["preset_prompt"]]    = preset
    wv[PROMOTED["keep_last_prompt"]] = keep_last
    wv[PROMOTED["steps"]]            = steps
    # 2) 内部副本双写（仅无歧义者）
    enh_id, preset_idx = ENHANCER[mode]
    enh = g_sg.nm[enh_id]["widgets_values"]
    enh[0] = model_name                        # model_name 恒在 widget[0]
    if preset_idx < len(enh):
        enh[preset_idx] = preset               # preset_prompt 按节点类型定位
    sched = g_sg.nm[9]["widgets_values"]        # BasicScheduler [scheduler, steps, denoise]
    if len(sched) > 1:
        sched[1] = steps
    return {"model_name": model_name, "preset_prompt": preset,
            "keep_last_prompt": keep_last, "steps": steps}


def inject_prompt(g_top, g_sg, args, mode, enhancer_bypassed):
    """把 --prompt 真正写进图里。**这是全链路唯一写 prompt 的地方。**

    此前无人写 prompt：apply_params 的 docstring 声明「prompt 属 task 7 不碰」，而 task 7 的
    rewire_core_prompt_to_raw 只把 core.prompt **重接**到子图边界，从没往任何地方写值。
    实测后果（/history 取服务端真实 API prompt 为证）：t2v/i2v/fl2va 渲染的是模板自带的
    vaporwave 片头示例（core 节点 widget[0] 里的那段 'Vaporwave title sequence look…'），
    r2v 渲染的是空串 —— 四模态全部与用户 prompt 无关。dry-run 只验结构，验不出这个。

    根因是 prompt 有**两个来源**且谁赢不确定：
      - core 节点自身的 prompt widget[0]（模板里预置了 vaporwave 示例文本）
      - core 节点的 prompt 输入槽（模板连到增强器；旁路后被重接到子图边界）
    故这里做确定性处理，写三处 + 断一边：
      1) core 节点 widget[0] = prompt —— 执行期真正被读取的值
      2) 实例 #105 promoted widget[0] = prompt —— 子图边界（--enhance 路经增强器）
      3) 增强器自身 prompt widget（t2v=prompt_text[5] / 其余=custom_prompt[6]）
      4) 增强器被旁路时**剪断 core.prompt 输入边**：只留 widget 一个来源，
         彻底消除 link-vs-widget 的歧义（--enhance 时保留该边，让增强结果喂进去）
    """
    core_id = 149 if mode == "r2v" else 104

    g_sg.nm[core_id]["widgets_values"][0] = args.prompt      # (1) 权威
    g_top.nm[105]["widgets_values"][0] = args.prompt         # (2) 边界

    eid, _ = ENHANCER[mode]                                  # (3) --enhance 路
    widx = PROMPT_WIDGET[mode]
    n = g_sg.nm.get(eid)
    if n is not None:
        wv = n.setdefault("widgets_values", [])
        while len(wv) <= widx:
            wv.append("")
        wv[widx] = args.prompt

    cut = None                                               # (4) 消除二源歧义
    if enhancer_bypassed:
        _idx, inp = g_sg.input_slot(core_id, "prompt")
        if inp and inp.get("link") is not None:
            cut = inp["link"]
            g_sg.cut_link(cut)
    return core_id, cut


def _set_mode(g, nid, mode):
    """把 g 作用域内 nid 节点的 mode 置为 mode（0 正常 / 4 bypass）。节点缺失返回 False。"""
    n = g.nm.get(nid)
    if n is None:
        return False
    n["mode"] = mode
    return True


def apply_degradation(g_top, g_sg, args, mode, run_id):
    """task 7：三项旁路/降级决策，全部用 ComfyUI 原生 bypass(mode=4) 落地。

    为何用 mode 而非剪线重连：子图 I/O 是虚拟边界节点（-10/-20，不在 nodes 里，边靠
    definitions.subgraphs[].inputs/outputs 的 linkIds 记账），跨界重连易把边界记账写坏；
    而 bypass 是 ComfyUI 图→prompt 转换期的原生语义——被旁路节点不执行，其每个输出按
    类型就近透传到同类型输入，天然给出「无该节点」的等价核心链。缺省即降级（HF/TRT 不可
    达、Sage 内核未必可编），显式 --enhance/--postprocess/--sage 恢复。返回决策字典。

    (a) QwenVL 增强链旁路（缺省）：增强器（ENHANCER[mode]）mode=4。增强器 STRING 输出
        按类型透传其 STRING 输入（子图边界的原始 prompt），故原始 prompt 直达主推理节点
        prompt 口 + 子图 output(easy showAnything 预览)，不触发 HF 模型下载。--enhance 恢复。
    (b) TensorRT 上采样 / RIFE 后处理旁路（缺省）：根图 POST_NODES 全部 mode=4。核心链
        子图 VIDEO 输出 → SaveVideo #92（link 194）不经后处理，旁路后仍出 h264+aac；缺
        .engine 时不再阻断。--postprocess 恢复（需已 build 好 TensorRT engine）。
    (c) SageAttention 降级（缺省）：PathchSageAttentionKJ mode=4，其 MODEL 输出按类型透传
        MODEL 输入 → 回退默认注意力（内核不可编时仍可跑）；ModelPatchTorchSettings 保留
        （仅设 torch 后端开关，无第三方内核依赖）。--sage 恢复。
    """
    dec = {}

    # (a) QwenVL 增强链
    enh_id = ENHANCER[mode][0]
    if args.enhance:
        _set_mode(g_sg, enh_id, MODE_ACTIVE)
        dec["enhancer"] = f"ACTIVE 增强器#{enh_id}（需 HF 可达，本机通常不满足）"
    else:
        ok = _set_mode(g_sg, enh_id, MODE_BYPASS)
        core_id = 149 if mode == "r2v" else 104
        rewired = rewire_core_prompt_to_raw(g_sg, mode, core_id) if ok else None
        dec["enhancer"] = (f"BYPASS 增强器#{enh_id} → 原始 prompt 直连核心#{core_id}"
                           f"(link{rewired})"
                           if ok else f"!! 增强器#{enh_id} 未找到，跳过")

    # (b) TensorRT 上采样 / RIFE 后处理
    #     ⚠️ 模板默认把 SaveVideo #92 置 mode=4（bypass），以 VHS_VideoCombine #126 为出片——
    #     但 VHS 的 IMAGE 输入来自 TRT 上采样链，缺 .engine 即断链。而本脚本的轮询/下载以
    #     SaveVideo 的 run_id 前缀（H3_<run_id>）为准（build_workflow 已写好）。故：始终激活
    #     SaveVideo #92（子图 VIDEO 输出直连 link194，不经 TRT，稳出 h264+aac），后处理链
    #     （POST_NODES）按 --postprocess 开关；缺省旁路。
    _set_mode(g_top, 92, MODE_ACTIVE)
    if args.upscale and not args.postprocess:
        # --upscale：只启用**超分侧**，插帧侧必须保持旁路。
        # 原因（实测）：模板引用的节点类型是 AutoRifeTensorrt / AutoLoadRifeTensorrtModel，
        # 而已装的 ComfyUI-Rife-Tensorrt 只提供 RifeTensorrt / LoadRifeTensorrtModel
        # （NODE_CLASS_MAPPINGS 里没有 Auto 变体）→ 激活它们会被服务端判 unknown node type
        # 直接拒掉整个 prompt。超分侧的 UpscalerTensorrt / LoadUpscalerTensorrtModel 名字
        # 与模板一致，装上 tensorrt 后即可用。
        # 链路：#105.IMAGE →(Rife 旁路，按类型透传)→ #127 Upscaler → #126 VHS；
        #       音频 #105.AUDIO 直连 #126.audio，故上采样产物同样带 aac 音轨。
        for nid in (UPSCALE_LOAD, UPSCALE_NODE, VHS_NODE):
            _set_mode(g_top, nid, MODE_ACTIVE)
        for nid in RIFE_NODES:
            _set_mode(g_top, nid, MODE_BYPASS)
        vhs = g_top.nm.get(VHS_NODE)
        if vhs is not None and isinstance(vhs.get("widgets_values"), dict):
            wv = vhs["widgets_values"]
            # 模板这三项都是按「有插帧 + webm」配的，旁路插帧后必须改：
            #   frame_rate 48 → 24：帧数没翻倍，48 会让成片快一倍
            #   format webm → h264-mp4：与全链路其余环节（SaveVideo/ffprobe 门禁）一致
            #   filename_prefix 带 video/ 子目录且不含 run_id → 改为扁平 + run_id，
            #     既能被 H3up_<run_id>* 取回，也保证并发不互相覆盖
            wv["frame_rate"] = 24
            wv["format"] = "video/h264-mp4"
            wv["filename_prefix"] = f"H3up_{run_id}"
            wv.pop("videopreview", None)          # 纯前端预览状态，含他人机器绝对路径
        dec["postprocess"] = (f"UPSCALE-ONLY 超分[{UPSCALE_LOAD},{UPSCALE_NODE}]+VHS[{VHS_NODE}] "
                              f"ACTIVE（2x, RealESRGAN_x4）| 插帧{RIFE_NODES} BYPASS"
                              f"（模板用 AutoRife*，已装包只有 Rife*，激活会被服务端拒）"
                              f" | VHS→24fps/h264-mp4/H3up_{run_id} | SaveVideo#92 亦激活")
    else:
        target = MODE_ACTIVE if args.postprocess else MODE_BYPASS
        hit = [nid for nid in POST_NODES if _set_mode(g_top, nid, target)]
        dec["postprocess"] = (f"{'ACTIVE' if args.postprocess else 'BYPASS'} "
                              f"后处理节点{hit}"
                              + ("（需 TensorRT engine）+ SaveVideo#92 亦激活" if args.postprocess
                                 else " → 仅核心 SaveVideo#92 链出片（模板默认的 #92 旁路已纠正）"))

    # (c) KJNodes SageAttention
    sage_ids = [n["id"] for n in g_sg.nodes
                if n.get("type") == "PathchSageAttentionKJ"]
    stgt = MODE_ACTIVE if args.sage else MODE_BYPASS
    for sid in sage_ids:
        _set_mode(g_sg, sid, stgt)
    dec["sage"] = (f"{'ACTIVE' if args.sage else 'BYPASS'} "
                   f"SageAttention{sage_ids}"
                   + ("" if args.sage else " → 回退默认注意力"))
    return dec


def build_workflow(args, run_id, seed, up):
    """按模态选四个新模板之一并加载，返回 (wf, mode, frames)。

    模板选择与已验证的 run_id 输出隔离（task 2）；权重名 NVFP4 → int4/mixed convrot
    的改写在此处调用 rewrite_weights（task 3，四模态通用）。其余节点级补丁按后续任务
    在此基础上填充（届时用 Graph 对本图手术）：
      - ✅ 权重名 NVFP4 → int4/mixed convrot（编码器 → qwen3vl…int4_convrot）... task 3
      - ✅ 新增/透传子图输入 model_name / preset_prompt / keep_last_prompt / steps . task 4
      - ✅ 尺寸/时长/帧数/步数/采样器/调度器/种子/shift/ref_image_size 参数化（实例
        promoted widgets 与内部节点双写）...................................... task 5
      - QwenVL 增强链旁路 + TensorRT 上采样/RIFE 后处理降级 + Sage 回退 ...... ✅ task 7
      - ✅ --lora NAME[:strength]（LoraLoaderModelOnly 接管 MODEL 出边）........ task 8
    """
    if args.frames:
        frames = snap_frames(args.frames)
    else:
        frames = snap_frames(round(args.seconds * 24))
    frames_direct = args.frames is not None

    wf = json.load(open(select_template(args.mode)))
    # 结构校验：四模板推理链均封装在子图实例 #105 + definitions.subgraphs[0]
    assert wf.get("definitions", {}).get("subgraphs"), \
        f"{TEMPLATES[args.mode]} 缺少子图定义，模板结构异常"

    # run_id 输出隔离（SaveVideo 92，四模板同 id）——保留已验证逻辑
    for n in wf["nodes"]:
        if n["type"] == "SaveVideo":
            n["widgets_values"][0] = f"H3_{run_id}"

    # ---- 权重名改写（task 3）：NVFP4 → int4/mixed convrot，四模态通用 ----
    sg = wf["definitions"]["subgraphs"][0]
    g_top = Graph(wf, wf["nodes"], wf["links"])
    g_sg = Graph(wf, sg["nodes"], sg["links"])
    rewrite_weights(g_top, g_sg, args.mode)

    # ---- 写入已上传输入到 LoadImage 节点（i2v/fl2va/r2v）：模板 widget 原为示例图名，
    #      改写成 upload_inputs 落定的 run_id 前缀远端 basename（结构 link 模板已连好）。
    #      未提供对应输入的 LoadImage 会被旁路——否则模板示例图名会让服务端校验拒掉整个
    #      prompt（r2v 只给 1 张 --ref-image 时实测必发）----
    _li_written, _li_bypassed = inject_load_images(g_top, up, args.mode)
    if _li_bypassed:
        print(f"LoadImage 旁路（未提供对应输入，避免模板示例图名被服务端拒绝）: "
              f"{_li_bypassed}", flush=True)

    # ---- 透传四模板独有的子图新增输入（task 4）：model_name/preset_prompt/
    #      keep_last_prompt/steps（实例 #105 promoted widgets 权威 + 无歧义内部副本双写）----
    passthrough_new_inputs(g_top, g_sg, args, args.mode)

    # ---- 参数化（task 5）：尺寸经 patch_size 在顶层重接线（--size）或改 ResolutionSelector
    #      #115（--aspect/--megapixels）；其余（时长/帧数/步数/采样器/调度器/种子/shift/
    #      ref_image_size）由 apply_params 写实例 promoted widgets + 子图内部双写。----
    patch_size(g_top, 105, "width", "height", args,
               widgets_node=g_top.nm[105], wv_w=1, wv_h=2)
    apply_params(g_top, g_sg, args, seed, frames, frames_direct, args.mode)

    # ---- 旁路/降级（task 7）：QwenVL 增强链旁路（原始 prompt 直达 sampler）、TensorRT
    #      上采样/RIFE 后处理旁路（仅核心 SaveVideo 链出片）、SageAttention 回退默认注意力。
    #      全部以 mode=4 bypass 落地，缺省启用降级，--enhance/--postprocess/--sage 恢复。----
    dec = apply_degradation(g_top, g_sg, args, args.mode, run_id)

    # ---- prompt 注入（必须在 apply_degradation 的重接线之后）：写 core 节点 prompt widget
    #      （执行期真正被读的值）+ 边界 + 增强器，并在旁路时剪断 core.prompt 输入边。
    #      此前无人写 prompt，四模态实际渲染的是模板自带 vaporwave 示例或空串，故带**硬断言**。
    _core_id, _cut = inject_prompt(g_top, g_sg, args, args.mode, enhancer_bypassed=not args.enhance)
    _p = g_sg.nm[_core_id]["widgets_values"][0]
    assert isinstance(_p, str) and _p.strip(), \
        f"prompt 未落到核心#{_core_id} widget[0]（={_p!r}）——拒绝提交"
    assert g_top.nm[105]["widgets_values"][0] == args.prompt, "prompt 未落到子图边界"
    if not args.enhance:
        _idx, _inp = g_sg.input_slot(_core_id, "prompt")
        assert not (_inp and _inp.get("link") is not None), \
            "增强器已旁路但 core.prompt 仍有输入边，prompt 来源仍有歧义"

    return wf, args.mode, frames, dec


def main():
    ap = argparse.ArgumentParser(
        description="MiniMax H3 全模态视频生成一键跑（T2V/I2V/R2V 自动选择）")
    g_in = ap.add_argument_group("模态输入（fl2va 与 ref2va 互斥）")
    g_in.add_argument("--prompt", required=True, help="提示词（R2V 用 <Picture i> 引用参考图）")
    g_in.add_argument("--image", help="首帧图（→I2V，fl2va）")
    g_in.add_argument("--last-image", help="末帧图（→I2V，fl2va）")
    g_in.add_argument("--ref-image", action="append", default=[], metavar="IMG",
                      help="参考图（→R2V，≤2，可重复；提示词里按顺序叫 <Picture 1..2>）")
    g_in.add_argument("--ref-video", action="append", default=[], metavar="VID",
                      help="不受支持：R2VA 工作流未接 ref_video（无 LoadVideo 节点），传入即报错")
    g_in.add_argument("--ref-audio", action="append", default=[], metavar="AUD",
                      help="不受支持：R2VA 工作流未接 ref_audio（无 LoadAudio 节点），传入即报错")
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
    # 四个新模板独有的子图新增输入（旧模板没有）。缺省保留模板原值，给出即覆盖。
    g_gen.add_argument("--model-name", default=None, metavar="NAME",
                       help="QwenVL 增强器模型名（子图新增输入；缺省沿用模板；增强链 task7 会旁路）")
    g_gen.add_argument("--preset-prompt", default=None, metavar="PRESET",
                       help="增强器预设风格 preset_prompt（子图新增输入；缺省沿用模板）")
    g_gen.add_argument("--keep-last-prompt", action=argparse.BooleanOptionalAction,
                       default=None,
                       help="复用上次增强提示词 keep_last_prompt（子图新增输入；缺省沿用模板）")
    g_gen.add_argument("--scheduler", default="simple",
                       help="调度器（默认 simple；参考多的提示词可试 beta/normal）")
    g_gen.add_argument("--seed", type=int, default=None, help="种子（默认随机）")
    g_gen.add_argument("--shift-video", type=float, default=None,
                       help="视频流 flow shift（节点默认12.0；给出时插入 SigmaShift 节点）")
    g_gen.add_argument("--shift-audio", type=float, default=None,
                       help="音频流 flow shift（节点默认3.0；与 --shift-video 可单给）")
    g_gen.add_argument("--ref-image-size", choices=["match", "max"], default="match",
                       help="R2V 参考图缩放：match=跟画布（快）；max=2048 短边（身份更保真，慢数倍）")
    g_gen.add_argument("--lora", default=None, metavar="NAME[:strength]",
                       help="追加 LoRA：turbo/larry 文件名自动接 Larry 专用 LoRA+双时钟 sampler；"
                            "其它文件使用 LoraLoaderModelOnly。strength 缺省 1.0，"
                            "如 --lora minimax_h3_turbo_v4_step600_ema.safetensors:1.0")
    g_gen.add_argument("--lora-backend", choices=["auto", "larry", "native"], default="auto",
                       help="LoRA 接法：auto 根据文件名含 turbo/larry 自动启用 Larry 专用双时钟节点；"
                            "larry 强制启用，native 使用普通 LoraLoaderModelOnly")
    g_gen.add_argument("--dit", choices=list(DIT_VARIANTS), default=None,
                       help="DiT 量化档位（缺省随 --host：5090→int8、4090→int4、kk14590→int8 ConvRot）。"
                            "int8/fp8 走 triton 加速核；int4/mixed 的 w4a4 段只能 eager 反量化。"
                            "换档前须确认目标机已下载对应权重——两个来源命名不同，见 DIT_VARIANTS")
    g_gen.add_argument("--clip", choices=list(CLIP_VARIANTS), default=None,
                       help="文本编码器档位（缺省随 --host：5090→int8、4090→int4、kk14590→NVFP4 15.7G）。"
                            "kk14590 的 NVFP4 encoder 可整块装入 24G 显存；旧 int8 encoder 不行")
    # ---- task 7 旁路/降级开关（缺省全部启用降级；显式恢复对应链）----
    g_gen.add_argument("--enhance", action="store_true",
                       help="启用 QwenVL 提示词增强链（缺省旁路：原始 prompt 直达 sampler；"
                            "增强器需 HF 模型可达，本机通常不满足）")
    g_gen.add_argument("--upscale", action="store_true",
                       help="启用 TensorRT 2x 超分（RealESRGAN_x4→2x）并经 VHS 出 h264-mp4，"
                            "额外产出 H3up_<run_id>*.mp4（原生 24fps 那份仍保留）。"
                            "插帧仍旁路：模板引用的 AutoRifeTensorrt 节点已装包未提供。"
                            f"要求画布每边在 {UPSCALE_DIM_MIN}~{UPSCALE_DIM_MAX} 内（引擎 shape 范围）")
    g_gen.add_argument("--postprocess", action="store_true",
                       help="启用完整后处理链（TensorRT 上采样 + RIFE 插帧 + VHS 二次封装）。"
                            "⚠️ 插帧节点名与已装包不匹配，当前会被服务端拒；想只要超分请用 --upscale")
    g_gen.add_argument("--sage", action="store_true",
                       help="启用 KJNodes SageAttention（缺省旁路：回退默认注意力；"
                            "内核在本机 cu128 未必可编）")
    g_run = ap.add_argument_group("运行")
    g_run.add_argument("--dry-run", action="store_true",
                       help="只做模态判定与模板选择并打印，不上传/不提交（自检）")
    g_run.add_argument("--out", default=None, help="本地输出路径（默认 ~/Downloads/h3_时间戳.mp4）")
    g_run.add_argument("--no-download", action="store_true", help="只跑不拉回")
    g_run.add_argument("--host", choices=list(HOSTS), default=DEFAULT_HOST,
                       help=f"目标机档位（缺省 {DEFAULT_HOST}）。同时决定 --dit/--clip 的缺省值与 SSH 目标")
    g_run.add_argument("--server", default=None, help="显式覆盖 SSH 目标（缺省取 --host 档位）")
    g_run.add_argument("--ssh-port", type=int, default=None,
                       help="显式覆盖 SSH 端口（缺省取 --host 档位）")
    args = ap.parse_args()

    # ---- 主机档位 → SSH 目标 + 量化档位缺省（显式 --server/--dit/--clip 优先） ----
    prof = HOSTS[args.host]
    if args.server is None:
        args.server = prof["srv"]
    if args.ssh_port is None:
        args.ssh_port = prof["port"]
    if args.dit is None:
        args.dit = prof["dit"]
    if args.clip is None:
        args.clip = prof["clip"]
    # 显存兜底告警：旧 int8 encoder 25.28G 在 24G 卡上会被迫 offload 到 CPU；
    # 本目标默认 NVFP4 encoder 约15.7G，可以整块驻显存。
    if args.clip == "int8" and prof["vram"] < 32:
        print(f"!! 警告：--clip int8（25.28G）在 {args.host} 档位（{prof['vram']}G）上"
              f"无法整块驻留显存，会 offload 到 CPU 并极慢；24G 卡应用 --clip int4",
              file=sys.stderr, flush=True)

    # --upscale 的引擎 shape 门禁：engine 按 min/opt/max = 256/512/1280 build，
    # 输入超出区间会在推理期报错。只有显式 --size 时能在提交前知道尺寸，故仅此时拦截；
    # 走 --aspect/--megapixels 时尺寸由服务端 ResolutionSelector 运行期算出，无法预判。
    if args.upscale and args.size:
        try:
            _w, _h = (int(v) for v in args.size.lower().split("x"))
        except Exception:
            _w = _h = None
        if _w and _h and not (UPSCALE_DIM_MIN <= min(_w, _h)
                              and max(_w, _h) <= UPSCALE_DIM_MAX):
            sys.exit(f"!! --upscale 要求画布每边在 {UPSCALE_DIM_MIN}~{UPSCALE_DIM_MAX} 内"
                     f"（超分 engine 的 shape 范围），你给的是 {_w}x{_h}。\n"
                     f"   要么改小画布，要么去掉 --upscale。")

    # ---- DiT / 编码器档位落到全局常量（build_workflow 用） ----
    global UNET_FL2VA, UNET_REF2VA, CLIP_QWEN
    UNET_FL2VA, UNET_REF2VA = DIT_VARIANTS[args.dit]
    CLIP_QWEN = CLIP_VARIANTS[args.clip]

    # ---- 模态校验 ----
    fl2va_inputs = bool(args.image or args.last_image)
    ref2va_inputs = bool(args.ref_image or args.ref_video or args.ref_audio)
    assert not (fl2va_inputs and ref2va_inputs), \
        "--image/--last-image（fl2va）与 --ref-*（ref2va）互斥，不能混用"
    # R2V 的参考能力**就是 2 张参考图**，这不是缺陷而是 MiniMaxH3-R2VA-Qwen3VL.json 的设计：
    # 子图边界只暴露 ref_images.ref_image_0 / ref_image_1；核心 #149 的 ref_image_2 /
    # ref_videos.ref_video_0 / ref_video_audios.ref_video_audio_0 / ref_audios.ref_audio_0
    # 全部 link=None，且整个工作流（子图 + 根图）里没有任何 LoadVideo / LoadAudio 节点，
    # 根图恰好 2 个 LoadImage(#114/#152)。脚本据此对齐，不接受超出模板的输入。
    if args.ref_video or args.ref_audio:
        sys.exit(
            "!! --ref-video / --ref-audio 不受支持：MiniMaxH3-R2VA-Qwen3VL.json 没有接这些输入\n"
            "   （核心 #149 的 ref_video_0 / ref_video_audio_0 / ref_audio_0 槽未连线，\n"
            "   工作流里也没有 LoadVideo / LoadAudio 节点）。\n"
            "   传入只会被上传后丢弃、产出忽略参考视频/音频的结果，故直接拒绝。\n"
            "   本工作流的参考能力：--ref-image，最多 2 张。"
        )
    if len(args.ref_image) > 2:
        sys.exit(f"!! --ref-image 最多 2 张：模板子图只暴露 ref_image_0 / ref_image_1，"
                 f"你给了 {len(args.ref_image)} 张。")
    if ref2va_inputs:
        args.mode = "r2v"            # 参考 → MiniMaxH3-R2VA
    elif args.last_image:
        args.mode = "fl2va"          # 首尾帧 → MiniMaxH3-FL2VA（含仅给末帧）
    elif args.image:
        args.mode = "i2v"            # 首帧 → MiniMaxH3-I2VA
    else:
        args.mode = "t2v"            # 纯文本 → MiniMaxH3-T2VA

    run_id = uuid.uuid4().hex[:8]
    seed = args.seed if args.seed is not None else random.randint(0, 2**53 - 1)

    if args.dry_run:
        tmpl = select_template(args.mode)
        wf, _, frames, dec = build_workflow(args, run_id, seed, {})   # 含 task3 权重改写
        sg = wf.get("definitions", {}).get("subgraphs", [])
        inst = next(n for n in wf["nodes"] if n["id"] == 105)
        sn = {n["id"]: n for n in sg[0]["nodes"]} if sg else {}
        print(f"[dry-run] mode={args.mode} template={tmpl.name} "
              f"nodes={len(wf['nodes'])} subgraphs={len(sg)} "
              f"run_id={run_id} seed={seed} frames={frames} dit={args.dit}", flush=True)
        _cid = 149 if args.mode == "r2v" else 104
        _pv = sn[_cid]["widgets_values"][0] if _cid in sn else None
        print(f"[dry-run] 落图 prompt（核心#{_cid} widget[0]，执行期真正被读的值）: "
              f"{(_pv or '')[:80]!r}", flush=True)
        # dry-run 也把构建结果落盘：此前 dry-run 不落盘，导致「提交的图里到底写了什么」
        # 无法离线核对 —— prompt 从未落图这个 bug 正是由此长期隐形。
        _dump = Path("/tmp/run_h3_dryrun.json")
        _dump.write_text(json.dumps(wf, ensure_ascii=False))
        print(f"[dry-run] 构建结果已落盘: {_dump}", flush=True)
        print(f"[dry-run] 权重名（NVFP4→{args.dit} / 编码器 {args.clip}"
              f"，主机档位 {args.host}）:", flush=True)
        print(f"  inst#105 unet/clip/vae_v/vae_a = "
              f"{inst['widgets_values'][5]} | {inst['widgets_values'][6]} | "
              f"{inst['widgets_values'][7]} | {inst['widgets_values'][8]}", flush=True)
        print(f"  sg UNETLoader#6={sn[6]['widgets_values'][0]}  "
              f"CLIPLoader#13={sn[13]['widgets_values'][0]} (type={sn[13]['widgets_values'][1]})  "
              f"VAE#11={sn[11]['widgets_values'][0]}  VAE#24={sn[24]['widgets_values'][0]}", flush=True)
        iv = inst["widgets_values"]
        enh_id, preset_idx = ENHANCER[args.mode]
        print("[dry-run] 子图新增输入透传（task4）:", flush=True)
        print(f"  promoted#105 model_name={iv[9]!r} preset_prompt={iv[10]!r} "
              f"keep_last_prompt={iv[11]!r} steps={iv[12]!r}", flush=True)
        print(f"  内部副本 enhancer#{enh_id}[0]={sn[enh_id]['widgets_values'][0]!r} "
              f"preset[{preset_idx}]={sn[enh_id]['widgets_values'][preset_idx]!r}  "
              f"BasicScheduler#9 steps={sn[9]['widgets_values'][1]!r}", flush=True)
        core_id = 149 if args.mode == "r2v" else 104
        core_wv = sn[core_id]["widgets_values"]
        idx_len, inp_len = (lambda ns: next(((i, x) for i, x in enumerate(ns)
                                             if x.get("name") == "length"), (None, None)))(
            sn[core_id].get("inputs", []))
        shift_ids = [n["id"] for n in sg[0]["nodes"] if n["type"] == "MiniMaxH3SigmaShift"]
        print("[dry-run] 参数化写入（task5）:", flush=True)
        print(f"  seed: inst#105[4]={iv[4]!r}  RandomNoise#15={sn[15]['widgets_values']!r}", flush=True)
        print(f"  steps: inst#105[12]={iv[12]!r}  sampler#17={sn[17]['widgets_values'][0]!r}  "
              f"scheduler#9={sn[9]['widgets_values'][0]!r}", flush=True)
        print(f"  时长/帧数: inst#105 duration[3]={iv[3]!r}  PrimitiveFloat#111={sn[111]['widgets_values']!r}  "
              f"帧数(吸附)={frames}  core#{core_id}.length_link="
              f"{(inp_len.get('link') if inp_len else None)!r} core.width/height/length="
              f"{core_wv[1]!r}/{core_wv[2]!r}/{core_wv[3]!r}", flush=True)
        rs = next((n for n in wf["nodes"] if n["id"] == 115), None)
        prim = [n for n in wf["nodes"] if n["type"] == "PrimitiveInt"]
        print(f"  尺寸: --size={args.size!r}  inst#105 w/h=[{iv[1]!r},{iv[2]!r}]  "
              f"ResolutionSelector#115={rs['widgets_values'] if rs else None!r}  "
              f"注入 PrimitiveInt={[n['widgets_values'] for n in prim]!r}", flush=True)
        print(f"  shift: --shift-video={args.shift_video!r} --shift-audio={args.shift_audio!r}  "
              f"SigmaShift 节点={shift_ids!r}", flush=True)
        # LoRA（task8）：确认 LoraLoaderModelOnly 已插入、强度已应用、UNET MODEL 出边已改道
        lora_nodes = [n for n in sg[0]["nodes"]
                      if n["type"] in ("LoraLoaderModelOnly", "MiniMaxH3TurboLoRA")]
        g_dbg = Graph(wf, sg[0]["nodes"], sg[0]["links"])   # 只为读 link 字段（格式无关）
        def _model_dsts(oid):   # 某节点 MODEL 出边落向的 (target_id, target_slot)
            return [(g_dbg._l(l, "target_id"), g_dbg._l(l, "target_slot"))
                    for l in sg[0]["links"]
                    if g_dbg._l(l, "origin_id") == oid and g_dbg._l(l, "type") == "MODEL"]
        if lora_nodes:
            ln = lora_nodes[0]
            print("[dry-run] LoRA 接线（task8）:", flush=True)
            print(f"  {ln['type']}#{ln['id']} widgets={ln['widgets_values']!r}",
                  flush=True)
            turbo_samplers = [n for n in sg[0]["nodes"]
                              if n["type"] == "MiniMaxH3TurboSampler"]
            print(f"  UNET#6 MODEL→{_model_dsts(6)!r}  LoRA#{ln['id']} MODEL→{_model_dsts(ln['id'])!r}"
                  f"  LarrySampler={[n['id'] for n in turbo_samplers]!r}",
                  flush=True)
        else:
            print(f"[dry-run] LoRA: --lora={args.lora!r}（未插入 LoraLoaderModelOnly）", flush=True)
        if args.mode == "r2v":
            print(f"  ref_image_size: #149[4]={sn[149]['widgets_values'][4]!r}", flush=True)
        # ---- 旁路/降级（task 7）：打印三项决策 + 校验核心链完整、无 HF/TRT 依赖 ----
        core_id = 149 if args.mode == "r2v" else 104
        enh_mode = sn[enh_id]["mode"]
        sage_ids = [n["id"] for n in sg[0]["nodes"] if n["type"] == "PathchSageAttentionKJ"]
        sage_modes = {sid: sn[sid]["mode"] for sid in sage_ids}
        post_modes = {nid: next((n["mode"] for n in wf["nodes"] if n["id"] == nid), None)
                      for nid in POST_NODES}
        active_root = {n["id"]: n["type"] for n in wf["nodes"]
                       if n.get("mode", 0) != 4 and n["type"] not in ("MarkdownNote",)}
        active_sg = {n["id"]: n["type"] for n in sg[0]["nodes"] if n.get("mode", 0) != 4}
        # 核心链必须完好且全部 active：UNET→(Sage 旁路后)→BasicGuider/BasicScheduler→
        # SamplerCustomAdvanced→VAEDecode(+Audio)→CreateVideo→子图 VIDEO→SaveVideo#92
        CORE_SG = [6, 9, 16, 14, 17, 15, 10, 23, 91, 11, 24, 13, core_id]
        core_ok = all(sn[c]["mode"] != 4 for c in CORE_SG if c in sn) \
            and active_root.get(92) == "SaveVideo"
        hf_dep = enh_mode != 4 and not args.enhance   # 增强器未旁路 = HF 依赖
        # TRT 依赖的判定要分情况：--upscale / --postprocess 时**有** TRT 依赖是预期结果，
        # 不该再断言为零。真正必须永远为零的是「插帧节点未旁路」——模板引用的
        # AutoRifeTensorrt / AutoLoadRifeTensorrtModel 在已装包里不存在，一旦激活整个
        # prompt 会被服务端判 unknown node type 拒掉。故拆成两个检查。
        trt_dep = any(post_modes[n] != 4 for n in (128, 127, 146, 145))
        trt_unexpected = trt_dep and not (args.postprocess or args.upscale)
        rife_active = any(post_modes[n] != 4 for n in RIFE_NODES)
        print("[dry-run] 旁路/降级（task7）:", flush=True)
        print(f"  (a) 增强器#{enh_id} mode={enh_mode}（4=bypass）→ {dec['enhancer']}", flush=True)
        print(f"  (b) 后处理 modes={post_modes}（4=bypass）→ {dec['postprocess']}", flush=True)
        print(f"  (c) SageAttention modes={sage_modes}（4=bypass）→ {dec['sage']}", flush=True)
        _exp = ("核心链=True，HF=False；TRT=True 属预期（已显式启用超分/后处理）"
                if (args.postprocess or args.upscale)
                else "核心链=True，HF/TRT 依赖=False")
        print(f"  核心链完整={core_ok}  HF依赖={hf_dep}  TRT依赖={trt_dep}  "
              f"插帧激活={rife_active}  （期望：{_exp}）", flush=True)
        print(f"  仍激活的根图节点={active_root}", flush=True)
        print(f"  仍激活的子图节点={active_sg}", flush=True)
        assert core_ok, "!! 核心 SaveVideo 链不完整或被旁路——task7 校验失败"
        assert not hf_dep, "!! 增强链未旁路，仍有 HF 依赖——task7 校验失败"
        assert not trt_unexpected, "!! 未显式启用却有 TensorRT 依赖——task7 校验失败"
        if not args.postprocess:
            assert not rife_active, \
                ("!! 插帧节点未旁路：模板用 AutoRifeTensorrt / AutoLoadRifeTensorrtModel，"
                 "已装 ComfyUI-Rife-Tensorrt 只提供 RifeTensorrt / LoadRifeTensorrtModel，"
                 "激活会被服务端判 unknown node type 拒掉整个 prompt")
        dump = Path(f"/tmp/run_h3_dryrun_{args.mode}.json")
        json.dump(wf, open(dump, "w"), ensure_ascii=False)
        print(f"[dry-run] 已 dump 补丁后工作流 → {dump}", flush=True)
        return

    # ---- 上传输入 ----
    # 只上传真正会被接线的文件：refvid/refaud 已在 main() 入口拒绝，不会走到这里。
    orig_refs = list(args.ref_image)
    files = []
    if args.image:
        files.append((args.image, "image"))
    if args.last_image:
        files.append((args.last_image, "last_image"))
    files += [(p, f"refimg{i}") for i, p in enumerate(args.ref_image)]
    up = upload_inputs(args, run_id, files)
    # 脚本内部用统一 tag 查名
    args.ref_image = [f"refimg{i}" for i in range(len(args.ref_image))]

    wf, mode, frames, dec = build_workflow(args, run_id, seed, up)
    dur = frames / 24
    shift_info = ""
    if args.shift_video is not None or args.shift_audio is not None:
        sv = args.shift_video if args.shift_video is not None else 12.0
        sa = args.shift_audio if args.shift_audio is not None else 3.0
        shift_info = f" shift=v{sv}/a{sa}"
    print(f"run_id={run_id} mode={mode} seed={seed} 帧数={frames}@24fps（{dur:.2f}s）"
          f" steps={args.steps} {args.sampler}/{args.scheduler}{shift_info}", flush=True)
    print(f"降级/旁路(task7): 增强器={dec['enhancer']} | 后处理={dec['postprocess']} | "
          f"注意力={dec['sage']}", flush=True)
    if frames > 362:
        print(f"!! {frames} 帧超出训练范围（124~362），画质自负", file=sys.stderr)
    if mode == "r2v":
        tags = [f"<Picture {i+1}>={Path(p).name}" for i, p in enumerate(orig_refs)]
        print("参考标签映射（提示词里用同名标签）:\n  " + "\n  ".join(tags), flush=True)

    # ---- 全部中间路径按 run_id 隔离：并发跑多发时不互相覆盖 ----
    # 此前这三处是固定路径（/tmp/run_h3_workflow.json、~/run_h3_workflow.json、
    # /tmp/run_h3.log），两个终端同时跑会：①后者覆盖前者的工作流 → comfy run 读到
    # 别人的图，自己的 H3_{run_id}* 永不出现、轮询到超时；②共用日志导致失败信号串台
    # （别人报错被判成自己失败）。ComfyUI 服务端本身是单 prompt_worker 串行排队，没问题，
    # 不安全的只是这层提交管道。
    wf_remote = f"~/run_h3_workflow_{run_id}.json"
    log_remote = f"/tmp/run_h3_{run_id}.log"
    rc_remote = f"~/.run_h3_{run_id}.rc"          # comfy run 的退出码哨兵，见下
    tmp = Path(f"/tmp/run_h3_workflow_{run_id}.json")
    json.dump(wf, open(tmp, "w"), ensure_ascii=False)
    scp_to(args.server, args.ssh_port, str(tmp), wf_remote)

    # ---- 提交执行 ----
    # 用 bash -c 包一层，让 comfy run 退出后把退出码写进本 run 专属的哨兵文件。
    # 完成判定改用「哨兵文件出现」而非 pgrep：pgrep 'comfy run' 会匹配到**别人**那一发，
    # 并发时会误判（别人没跑完就以为自己没完，别人退出就以为自己完了）。
    ssh(args.server, args.ssh_port,
        f"export PATH=$HOME/h3-venv/bin:$HOME/.local/bin:$PATH; rm -f {rc_remote}; "
        f"setsid nohup bash -c 'comfy run --workflow {wf_remote} "
        f"--host 127.0.0.1 --port 8188 --wait --verbose --timeout 10800 "
        f"> {log_remote} 2>&1; echo $? > {rc_remote}' </dev/null >/dev/null 2>&1 & "
        "disown; echo submitted")
    print("已提交，轮询产物...", flush=True)

    # ---- 轮询产物并拉回 ----
    # 完成信号 = 本 run 的退出码哨兵出现，且**以其数值为权威**（文件出现≠写完，moov 最后才落盘）。
    t0 = time.time()
    out_glob = f"{SVR_OUTPUT}/H3_{run_id}*"
    # 三态一次取回（本 run 日志失败标记 / 本 run 退出码 / 服务进程），把每轮 3 次 ssh 压成 1 次。
    # 前两项都只看本 run 专属文件，与并发的其他 run 完全隔离；只有 ComfyUI 存活检查是全局的
    # （合理：服务挂了对谁都是致命的）。pgrep 用方括号打断自匹配。
    # rc 取值：'-'=还没结束；'0'=成功；其它=comfy run 非零退出。
    # 保留日志 grep 是因为它能给出**可读的错误原文**，而 rc 只给一个数字；两者互补：
    # grep 负责「为什么失败」，rc 负责「到底算不算失败」——只靠 grep 会漏掉那些既不打印
    # "ok": false 也不打印 Traceback 就非零退出的情况（那时旧逻辑会误判成「成功但找不到产物」）。
    probe = (f"grep -qE '\"ok\": false|Traceback' {log_remote} 2>/dev/null && s=FAIL || s=OK; "
             f"if [ -f {rc_remote} ]; then r=$(cat {rc_remote} 2>/dev/null || echo 9); else r=-; fi; "
             "pgrep -f '[m]ain.py' >/dev/null && a=y || a=n; "
             "echo \"$s $r $a\"")
    while True:
        time.sleep(20)
        if time.time() - t0 > 10800:
            print("!! 轮询超时（3h）", file=sys.stderr)
            sys.exit(5)
        st = ssh(args.server, args.ssh_port, probe, capture=True, retries=6).stdout.split()
        if len(st) != 3:
            print(f"  !! 探针输出异常 {st!r}，本轮跳过", file=sys.stderr, flush=True)
            continue
        logfail, rc, alive = st
        if logfail == "FAIL":
            err = ssh(args.server, args.ssh_port,
                      f"tail -15 {log_remote}", capture=True).stdout
            print("!! 执行失败\n", err, file=sys.stderr)
            if not args.no_download:
                salvage_core_output(args.server, args.ssh_port, out_glob,
                                    run_id, mode)
            sys.exit(3)
        if rc != "-":                      # comfy run 已结束，以退出码为准
            if rc != "0":
                err = ssh(args.server, args.ssh_port,
                          f"tail -15 {log_remote}", capture=True).stdout
                print(f"!! comfy run 非零退出（rc={rc}）\n", err, file=sys.stderr)
                if not args.no_download:
                    salvage_core_output(args.server, args.ssh_port, out_glob,
                                        run_id, mode)
                sys.exit(3)
            break                          # rc==0 → 真正成功，去取产物
        if alive != "y":
            err = ssh(args.server, args.ssh_port,
                      f"tail -8 {log_remote}", capture=True).stdout
            print("!! ComfyUI 进程消失\n", err, file=sys.stderr)
            sys.exit(2)
        print(f"  ... {int(time.time()-t0)}s", flush=True)

    cur = ssh(args.server, args.ssh_port,
              f"ls -t {out_glob} 2>/dev/null | head -1",
              capture=True, retries=6).stdout.strip()
    if not cur:
        err = ssh(args.server, args.ssh_port,
                  f"tail -15 {log_remote}", capture=True).stdout
        print("!! 流程结束但找不到产物\n", err, file=sys.stderr)
        sys.exit(4)
    print("产物就绪:", cur, flush=True)

    if not args.no_download:
        out = args.out or str(Path.home() / "Downloads" /
                              f"h3_{mode}_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
        ok, out, probe = pull_output(args.server, args.ssh_port, out_glob,
                                     out, run_id, "core")
        if not ok:
            print("!! 产物仍不可读，服务器原件保留在 " + cur, file=sys.stderr)
            sys.exit(4)
        if "codec_type=audio" not in probe.stdout:
            print("!! 警告：产物没有音轨（H3 应原生出声）", file=sys.stderr)
        ssh(args.server, args.ssh_port, f"rm -f {SVR_INPUT}/h3_{run_id}_*")
        print(probe.stdout.strip())
        print("DONE ->", out)

        # --upscale 时 VHS 会另出一份 2x 产物（H3up_<run_id>*），一并拉回。
        # 原生那份不丢：两者互为对照，也便于确认超分链是否真的生效。
        if args.upscale:
            up_glob = f"{SVR_OUTPUT}/H3up_{run_id}*"
            up_out = str(Path(out).with_name(Path(out).stem + "_2x" + Path(out).suffix))
            ok_up, up_out, up_probe = pull_output(args.server, args.ssh_port,
                                                  up_glob, up_out, run_id, "up")
            if not ok_up:
                print("!! --upscale 指定了但没找到/不可读 H3up_ 产物，超分链可能没执行；"
                      f"查 {log_remote}", file=sys.stderr)
            else:
                if "codec_type=audio" not in up_probe.stdout:
                    print("!! 警告：超分产物没有音轨", file=sys.stderr)
                print(up_probe.stdout.strip())
                print("DONE(2x) ->", up_out)
    else:
        print("DONE (no-download)")


if __name__ == "__main__":
    main()
