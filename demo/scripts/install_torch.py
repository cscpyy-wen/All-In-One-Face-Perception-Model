"""Download PyTorch wheels with progress bar, then install from local files."""
import sys
import subprocess
import os
import tempfile
from urllib.request import urlretrieve

MIRROR = "https://pypi.tuna.tsinghua.edu.cn/packages"
PACKAGES = {
    "torch": "b9/dc/1f1f621afe15e3c496e1e8f94f8903f75f87e7d642d5a985e92210cc208d/torch-2.8.0-cp39-cp39-win_amd64.whl",
}

def reporthook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = downloaded / total_size * 100
        bar_len = 40
        filled = int(bar_len * pct / 100)
        bar = '█' * filled + '░' * (bar_len - filled)
        mb_down = downloaded / 1048576
        mb_total = total_size / 1048576
        print(f"\r  [{bar}] {pct:5.1f}%  {mb_down:.1f}/{mb_total:.1f} MB", end='', flush=True)
    else:
        mb_down = downloaded / 1048576
        print(f"\r  {mb_down:.1f} MB downloaded", end='', flush=True)

def main():
    tmpdir = tempfile.mkdtemp(prefix="torch_install_")
    print(f"Download dir: {tmpdir}\n")

    downloaded_files = []
    for name, rel_path in PACKAGES.items():
        url = f"{MIRROR}/{rel_path}"
        filename = os.path.basename(rel_path)
        local_path = os.path.join(tmpdir, filename)
        print(f"Downloading {name}...")
        try:
            urlretrieve(url, local_path, reporthook)
            print("\n  Done!")
            downloaded_files.append(local_path)
        except Exception as e:
            print(f"\n  Error: {e}")
            # Fallback: just pip install
            print("\nFallback to pip install...")
            subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision",
                          "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"], check=False)
            return

    print("\n\nInstalling downloaded wheels...")
    cmd = [sys.executable, "-m", "pip", "install"] + downloaded_files + ["--force-reinstall"]
    print(" ".join(cmd))
    subprocess.run(cmd, check=False)

    # Also install torchvision
    print("\nInstalling torchvision...")
    subprocess.run([sys.executable, "-m", "pip", "install", "torchvision",
                   "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"], check=False)

    # Verify
    try:
        import torch
        print(f"\n✓ torch {torch.__version__} installed successfully!")
        print(f"  CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("\n✗ torch import failed")

if __name__ == "__main__":
    main()
