#!/usr/bin/env python3
"""多线程断点续传下载器，用于从 hf-mirror.com 拉取大模型文件。"""
import concurrent.futures as cf
import os
import subprocess
import sys
import time

CHUNK = 32 * 1024 * 1024  # 32MB 分片
WORKERS = 16

def head_size(url):
    out = subprocess.run(
        ["curl", "-sIL", "--max-time", "60", url],
        capture_output=True, text=True).stdout
    sizes = [int(l.split()[1]) for l in out.splitlines()
             if l.lower().startswith("content-length") and l.split()[1].isdigit()]
    return sizes[-1] if sizes else 0

def fetch_chunk(url, start, end, path, tries=30):
    for attempt in range(tries):
        r = subprocess.run(
            ["curl", "-sL", "--max-time", "300", "-r", f"{start}-{end}",
             "-o", path],
            capture_output=True)
        if r.returncode == 0 and os.path.getsize(path) == end - start + 1:
            return True
        time.sleep(min(2 * (attempt + 1), 30))
    return False

def download(url, dest):
    tmpdir = dest + ".parts"
    os.makedirs(tmpdir, exist_ok=True)
    size = head_size(url)
    assert size > 0, f"cannot get size of {url}"
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        print(f"[skip] {dest} already complete")
        return True
    ranges = [(s, min(s + CHUNK, size) - 1)
              for s in range(0, size, CHUNK)]
    print(f"[start] {os.path.basename(dest)}: {size/1e9:.2f} GB, {len(ranges)} chunks")
    t0 = time.time()
    ok_all = True
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        futs = {}
        for i, (s, e) in enumerate(ranges):
            part = os.path.join(tmpdir, f"{i:05d}")
            if os.path.exists(part) and os.path.getsize(part) == e - s + 1:
                continue
            futs[ex.submit(fetch_chunk, url, s, e, part)] = i
        n_done = 0
        for f in cf.as_completed(futs):
            n_done += 1
            if not f.result():
                ok_all = False
                print(f"[FAIL] chunk {futs[f]}")
            if n_done % 20 == 0 or n_done == len(futs):
                mb = sum(os.path.getsize(os.path.join(tmpdir, p))
                         for p in os.listdir(tmpdir) if p != "DONE") / 1e6
                dt = time.time() - t0
                print(f"  {mb/1e3:.2f}/{size/1e9:.2f} GB  {mb/dt/1000:.2f} MB/s", flush=True)
    if not ok_all:
        return False
    with open(dest, "wb") as out:
        for i in range(len(ranges)):
            part = os.path.join(tmpdir, f"{i:05d}")
            with open(part, "rb") as p:
                out.write(p.read())
            os.remove(part)
    os.rmdir(tmpdir)
    assert os.path.getsize(dest) == size, "size mismatch after merge"
    print(f"[done] {dest} in {(time.time()-t0)/60:.1f} min")
    return True

if __name__ == "__main__":
    jobs = [
        ("https://hf-mirror.com/Comfy-Org/Qwen-Image-2.1/resolve/main/diffusion_models/qwen_image_2.1_int8_convrot.safetensors",
         os.path.expanduser("~/qwen-image21/ComfyUI/models/diffusion_models/qwen_image_2.1_int8_convrot.safetensors")),
        ("https://hf-mirror.com/Comfy-Org/Qwen-Image-2.1/resolve/main/text_encoders/qwen3vl_8b_int8_convrot.safetensors",
         os.path.expanduser("~/qwen-image21/ComfyUI/models/text_encoders/qwen3vl_8b_int8_convrot.safetensors")),
        ("https://hf-mirror.com/Comfy-Org/Qwen-Image-2.1/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors",
         os.path.expanduser("~/qwen-image21/ComfyUI/models/vae/qwen_image_2.1_vae_bf16.safetensors")),
    ]
    rc = 0
    for url, dest in jobs:
        if not download(url, dest):
            rc = 1
    sys.exit(rc)
