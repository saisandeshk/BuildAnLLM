"""System info utilities for backend endpoints."""

from __future__ import annotations

import os
import platform
import subprocess
from typing import Dict

import psutil
import torch


def _get_nice_cpu_name() -> str:
    system = platform.system()
    if system == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass
    elif system == "Windows":
        return platform.processor()
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return platform.processor() or platform.machine() or "Unknown CPU"


def _get_os_name() -> str:
    def _get_mac_os_codename(version: str) -> str:
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
        for key, codename in codename_map.items():
            if version.startswith(key):
                return codename
        return ""

    system = platform.system()
    if system == "Darwin":
        try:
            product_version = subprocess.check_output(
                ["sw_vers", "-productVersion"]
            ).decode().strip()
            codename = _get_mac_os_codename(product_version)
            if codename:
                return f"macOS {codename} {product_version}"
            return f"macOS {product_version}"
        except Exception:
            return "macOS (version unknown)"

    return f"{system} {platform.release()}"


def get_system_info() -> Dict[str, str]:
    cpu_info = _get_nice_cpu_name()
    num_threads = torch.get_num_threads()
    total_cores = os.cpu_count() or 0
    mem_gb = psutil.virtual_memory().total / (1024 ** 3)
    os_info = _get_os_name()
    python_version = platform.python_version()
    torch_version = torch.__version__

    if torch.cuda.is_available():
        gpu = f"CUDA ({torch.cuda.get_device_name(0)})"
    else:
        gpu = "Not available"

    mps_available = bool(
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    )
    mps = "Available via Apple Silicon" if mps_available else "Not available"

    return {
        "cpu": f"{cpu_info} with {num_threads} threads/{total_cores} cores",
        "ram_gb": f"{mem_gb:.0f}",
        "gpu": gpu,
        "mps": mps,
        "os": os_info,
        "python": python_version,
        "torch": torch_version,
    }

