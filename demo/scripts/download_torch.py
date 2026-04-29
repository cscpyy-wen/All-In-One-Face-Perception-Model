"""Download PyTorch CUDA wheel manually, then install."""
import os
import sys
import subprocess
import urllib.request
from pathlib import Path

TORCH_URL = "https://download-r2.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp39-cp39-win_amd64.whl"
TORCHVISION_URL = "https://download-r2.pytorch.org/whl/cu124/torchvision-0.21.0%2Bcu124-cp39-cp39-win_amd64.whl"
SAVE_DIR = str(Path(__file__).resolve().parent.parent.parent / "_wheels")

def reporthook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = downloaded / total_size * 100
        mb_down = downloaded / 1048576
        mb_total = total_size / 1048576
        bar_len = 40
        filled = int(bar_len * min(pct, 100) / 100)
        bar = '█' * filled + '░' * (bar_len - filled)
        sys.stdout.write(f"\r  [{bar}] {pct:5.1f}%  {mb_down:.0f}/{mb_total:.0f} MB")
        sys.stdout.flush()

def download_with_resume(url, filepath):
    """Download with resume support."""
    tmp_path = filepath + ".tmp"
    downloaded = 0
    if os.path.exists(tmp_path):
        downloaded = os.path.getsize(tmp_path)
        print(f"  Resuming from {downloaded / 1048576:.0f} MB")

    req = urllib.request.Request(url)
    if downloaded > 0:
        req.add_header('Range', f'bytes={downloaded}-')

    response = urllib.request.urlopen(req)
    total_size = int(response.headers.get('content-length', 0)) + downloaded

    mode = 'ab' if downloaded > 0 else 'wb'
    with open(tmp_path, mode) as f:
        while True:
            chunk = response.read(8192)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            reporthook(0, 1, total_size)  # Update progress

    print()
    os.rename(tmp_path, filepath)
    print(f"  Saved: {filepath}")

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    torch_whl = os.path.join(SAVE_DIR, "torch-2.6.0+cu124-cp39-cp39-win_amd64.whl")
    torchvision_whl = os.path.join(SAVE_DIR, "torchvision-0.21.0+cu124-cp39-cp39-win_amd64.whl")

    if not os.path.exists(torch_whl):
        print("Downloading torch (2.5 GB)...")
        download_with_resume(TORCH_URL, torch_whl)
    else:
        print(f"torch already downloaded: {torch_whl}")

    if not os.path.exists(torchvision_whl):
        print("Downloading torchvision...")
        download_with_resume(TORCHVISION_URL, torchvision_whl)
    else:
        print(f"torchvision already downloaded: {torchvision_whl}")

    print("\nInstalling from local wheels...")
    subprocess.run([sys.executable, "-m", "pip", "install",
                   torch_whl, torchvision_whl, "--force-reinstall"], check=True)

    import torch
    print(f"\ntorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")

if __name__ == "__main__":
    main()
