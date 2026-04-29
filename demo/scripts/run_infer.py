"""Wrapper to set environment variables before running inference."""
import os
import sys

# Set CUDA_HOME to conda's CUDA toolkit
conda_prefix = os.path.dirname(sys.executable)
cuda_home = os.path.join(conda_prefix, "Library")
os.environ["CUDA_HOME"] = cuda_home
os.environ["CUDA_PATH"] = cuda_home

# Add CUDA bin to PATH
cuda_bin = os.path.join(cuda_home, "bin")
os.environ["PATH"] = cuda_bin + ";" + os.environ.get("PATH", "")

# Set MSVC environment
vs_path = r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools"
msvc_path = os.path.join(vs_path, "VC", "Tools", "MSVC", "14.50.35717")
msvc_bin = os.path.join(msvc_path, "bin", "Hostx64", "x64")
msvc_include = os.path.join(msvc_path, "include")
msvc_lib = os.path.join(msvc_path, "lib", "x64")

os.environ["PATH"] = msvc_bin + ";" + os.environ.get("PATH", "")
os.environ["INCLUDE"] = msvc_include + ";" + os.environ.get("INCLUDE", "")
os.environ["LIB"] = msvc_lib + ";" + os.environ.get("LIB", "")

# Windows SDK
winsdk_include = r"C:\Program Files (x86)\Windows Kits\10\Include"
winsdk_lib = r"C:\Program Files (x86)\Windows Kits\10\Lib"
if os.path.isdir(winsdk_include):
    # Find the latest SDK version
    sdk_versions = [d for d in os.listdir(winsdk_include) if d.startswith("10.")]
    if sdk_versions:
        latest_sdk = sorted(sdk_versions)[-1]
        os.environ["INCLUDE"] = os.path.join(winsdk_include, latest_sdk, "ucrt") + ";" + os.environ["INCLUDE"]
        os.environ["INCLUDE"] = os.path.join(winsdk_include, latest_sdk, "um") + ";" + os.environ["INCLUDE"]
        os.environ["INCLUDE"] = os.path.join(winsdk_include, latest_sdk, "shared") + ";" + os.environ["INCLUDE"]
        ucrt_lib = os.path.join(winsdk_lib, latest_sdk, "ucrt", "x64")
        um_lib = os.path.join(winsdk_lib, latest_sdk, "um", "x64")
        os.environ["LIB"] = ucrt_lib + ";" + os.environ["LIB"]
        os.environ["LIB"] = um_lib + ";" + os.environ["LIB"]

print(f"CUDA_HOME: {os.environ.get('CUDA_HOME')}")
print(f"CUDA_PATH: {os.environ.get('CUDA_PATH')}")

# Now run the actual inference script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Moeface_project"))
exec(open(os.path.join(os.path.dirname(__file__), "infer.py"), encoding="utf-8").read())
