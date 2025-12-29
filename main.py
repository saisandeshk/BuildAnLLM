
import os
import platform
import re
import subprocess
import threading
from collections import deque
from pathlib import Path

import psutil
import streamlit as st
import torch

from utils import get_device, scan_checkpoints
from pretraining.model.model_loader import load_model_from_checkpoint


st.set_page_config(
    page_title="Transformer Training & Inference",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "training_active" not in st.session_state:
    st.session_state.training_active = False
if "trainer" not in st.session_state:
    st.session_state.trainer = None
if "training_thread" not in st.session_state:
    st.session_state.training_thread = None
if "loss_data" not in st.session_state:
    st.session_state.loss_data = {
        "iterations": [], "train_losses": [], "val_losses": []}
if "training_logs" not in st.session_state:
    st.session_state.training_logs = []
if "current_model" not in st.session_state:
    st.session_state.current_model = None
if "current_tokenizer" not in st.session_state:
    st.session_state.current_tokenizer = None
if "shared_loss_data" not in st.session_state:
    st.session_state.shared_loss_data = {
        "iterations": [], "train_losses": [], "val_losses": []}
if "shared_training_logs" not in st.session_state:
    st.session_state.shared_training_logs = deque(
        maxlen=200)  # Thread-safe deque
if "training_lock" not in st.session_state:
    st.session_state.training_lock = threading.Lock()





st.markdown("### 🖥️ Your Device")

# --- CPU Info ---


def get_nice_cpu_name():
    system = platform.system()
    if system == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True
            )
            return result.stdout.strip()
        except Exception:
            pass
    elif system == "Windows":
        return platform.processor()
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        except Exception:
            pass
    return platform.processor() or platform.machine() or "Unknown CPU"

# --- macOS Version Mapping ---


def get_os_name():
    def get_mac_os_codename(version: str) -> str:
        codename_map = {
            "15": "Sequoia",
            "14": "Sonoma",
            "13": "Ventura",
            "12": "Monterey",
            "11": "Big Sur",
            "10.15": "Catalina",
            "10.14": "Mojave",
            "10.13": "High Sierra",
            "10.12": "Sierra",
            "10.11": "El Capitan",
            "10.10": "Yosemite",
        }
        for key in codename_map:
            if version.startswith(key):
                return codename_map[key]
        return ""

    system = platform.system()
    if system == "Darwin":
        try:
            product_version = subprocess.check_output(
                ["sw_vers", "-productVersion"]
            ).decode().strip()
            codename = get_mac_os_codename(product_version)
            return f"macOS {codename} {product_version}" if codename else f"macOS {product_version}"
        except Exception:
            return "macOS (version unknown)"
    return f"{system} {platform.release()}"


# --- Gather All Info ---
cpu_info = get_nice_cpu_name()
num_threads = torch.get_num_threads()
total_cores = os.cpu_count()
mem_gb = psutil.virtual_memory().total / (1024 ** 3)
os_info = get_os_name()
python_version = platform.python_version()
torch_version = torch.__version__

# --- GPU Info ---
if torch.cuda.is_available():
    gpu = f"CUDA ({torch.cuda.get_device_name(0)})"
else:
    gpu = "Not available (Try [RunPod](https://runpod.io?ref=avnw83xb))"

mps = "Available via Apple Silicon" if getattr(torch.backends, "mps",
                                               None) and torch.backends.mps.is_available() else "Not available"

# --- Display as Markdown Table ---
st.markdown(f"""
| Component        | Details                                |
|------------------|-----------------------------------------|
| **CPU**          | {cpu_info} — {num_threads} threads / {total_cores} cores |
| **RAM**          | {mem_gb:.2f} GB                         |
| **GPU (CUDA)**   | {gpu}                                   |
| **MPS**          | {mps}                                   |
| **OS**           | {os_info}                               |
| **Python**       | {python_version}                        |
| **PyTorch**      | {torch_version}                         |
""")

with open("README.md", encoding="utf-8") as f:
    content = f.read()
    content = re.sub(r'<img[^>]*>', '', content)
    st.markdown(content)

# Store helper functions in session state for access by pages
if "get_device" not in st.session_state:
    st.session_state.get_device = get_device
if "scan_checkpoints" not in st.session_state:
    st.session_state.scan_checkpoints = scan_checkpoints
if "load_model_from_checkpoint" not in st.session_state:
    st.session_state.load_model_from_checkpoint = load_model_from_checkpoint