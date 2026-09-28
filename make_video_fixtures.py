"""视频库音频 → 16k mono f32 夹具（对齐测试输入）。

扫描 C:\\Users\\ADMIN\\Videos 下全部 *.wav / *.mp3（递归），
ffmpeg 转 16k mono f32le → wgpu/ref/video/vNN.f32，
并写 manifest.json（安全名 → 原始路径 + 时长）。
"""

import json
import os
import subprocess
import sys

SRC_DIR = r"C:\Users\ADMIN\Videos"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref", "video")
EXTS = (".wav", ".mp3")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = []
    for root, _, names in os.walk(SRC_DIR):
        for n in names:
            if n.lower().endswith(EXTS):
                files.append(os.path.join(root, n))
    files.sort()
    print(f"found {len(files)} audio files")

    manifest = []
    for i, src in enumerate(files, 1):
        name = f"v{i:02d}"
        out = os.path.join(OUT_DIR, f"{name}.f32")
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", src, "-ac", "1", "-ar", "16000",
            "-f", "f32le", "-acodec", "pcm_f32le", out,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  {name} FAIL {src}: {r.stderr.strip()[:200]}")
            continue
        dur = os.path.getsize(out) / 4 / 16000
        manifest.append({"name": name, "source": src, "dur_s": round(dur, 2)})
        print(f"  {name}  {dur:8.1f}s  {src}")
    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    total = sum(m["dur_s"] for m in manifest)
    print(f"total {len(manifest)} fixtures, {total:.0f}s audio")


if __name__ == "__main__":
    main()
