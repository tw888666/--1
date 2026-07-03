# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2026 ECO Authors. All rights reserved.

import os
import subprocess
import sys


def _get_arg_value(argv, name, default):
    prefix = f"{name}="
    for idx, arg in enumerate(argv):
        if arg.startswith(prefix):
            return arg[len(prefix) :]
        if arg == name and idx + 1 < len(argv):
            return argv[idx + 1]
    return default


def _has_flag(argv, name):
    return name in argv


def _visible_cuda_device_ordinals():
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_devices:
        return None

    physical_ids = []
    for item in visible_devices.split(","):
        item = item.strip()
        if not item:
            continue
        if not item.isdigit():
            raise RuntimeError(
                "Auto GPU selection supports only numeric CUDA_VISIBLE_DEVICES "
                f"entries, got: {visible_devices}"
            )
        physical_ids.append(int(item))

    if not physical_ids:
        return None
    return {physical_id: ordinal for ordinal, physical_id in enumerate(physical_ids)}


def query_gpu_status():
    query = "index,memory.total,memory.used,memory.free,utilization.gpu"
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        encoding="utf-8",
        stderr=subprocess.STDOUT,
    )

    gpus = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            index, total, used, free, utilization = [int(float(part)) for part in parts]
        except ValueError:
            continue
        gpus.append(
            {
                "index": index,
                "memory_total_mb": total,
                "memory_used_mb": used,
                "memory_free_mb": free,
                "utilization_gpu": utilization,
            }
        )

    if not gpus:
        raise RuntimeError("nvidia-smi returned no usable GPU status rows.")
    return gpus


def select_idle_gpu(min_free_memory_mb=4096, max_utilization=20):
    visible_ordinals = _visible_cuda_device_ordinals()
    gpus = query_gpu_status()
    if visible_ordinals is not None:
        gpus = [gpu for gpu in gpus if gpu["index"] in visible_ordinals]
        if not gpus:
            raise RuntimeError(
                "No NVIDIA GPUs from nvidia-smi match CUDA_VISIBLE_DEVICES."
            )

    candidates = [
        gpu
        for gpu in gpus
        if gpu["memory_free_mb"] >= min_free_memory_mb
        and gpu["utilization_gpu"] <= max_utilization
    ]
    if not candidates:
        candidates = gpus
        print(
            "No GPU met the idle thresholds; selecting the GPU with the most "
            "free memory instead."
        )

    selected = max(
        candidates,
        key=lambda gpu: (
            gpu["memory_free_mb"],
            -gpu["utilization_gpu"],
            gpu["memory_total_mb"],
        ),
    )
    selected["cuda_device_id"] = (
        visible_ordinals[selected["index"]]
        if visible_ordinals is not None
        else selected["index"]
    )
    return selected


def selected_gpu_from_env():
    if os.environ.get("BRUCE_AUTO_GPU_SELECTION_APPLIED") != "1":
        return None

    try:
        index = int(os.environ["BRUCE_AUTO_SELECTED_GPU_INDEX"])
        total = int(os.environ["BRUCE_AUTO_SELECTED_GPU_TOTAL_MB"])
        free = int(os.environ["BRUCE_AUTO_SELECTED_GPU_FREE_MB"])
        utilization = int(os.environ["BRUCE_AUTO_SELECTED_GPU_UTILIZATION"])
    except (KeyError, ValueError):
        return None

    return {
        "index": index,
        "memory_total_mb": total,
        "memory_free_mb": free,
        "utilization_gpu": utilization,
        "cuda_device_id": 0,
    }


def apply_auto_gpu_selection_from_argv(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not _has_flag(argv, "--auto_select_gpu"):
        return None
    if os.environ.get("BRUCE_AUTO_GPU_SELECTION_APPLIED") == "1":
        return selected_gpu_from_env()

    min_free_memory_mb = int(_get_arg_value(argv, "--auto_gpu_min_free_memory_mb", 4096))
    max_utilization = int(_get_arg_value(argv, "--auto_gpu_max_utilization", 20))
    selected = select_idle_gpu(
        min_free_memory_mb=min_free_memory_mb,
        max_utilization=max_utilization,
    )

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected["index"])
    os.environ["BRUCE_AUTO_GPU_SELECTION_APPLIED"] = "1"
    os.environ["BRUCE_AUTO_SELECTED_GPU_INDEX"] = str(selected["index"])
    os.environ["BRUCE_AUTO_SELECTED_GPU_TOTAL_MB"] = str(selected["memory_total_mb"])
    os.environ["BRUCE_AUTO_SELECTED_GPU_FREE_MB"] = str(selected["memory_free_mb"])
    os.environ["BRUCE_AUTO_SELECTED_GPU_UTILIZATION"] = str(
        selected["utilization_gpu"]
    )
    selected["cuda_device_id"] = 0

    print(
        "Auto-selected GPU "
        f"{selected['index']} and set CUDA_VISIBLE_DEVICES={selected['index']} "
        f"(free {selected['memory_free_mb']} MiB / "
        f"{selected['memory_total_mb']} MiB, "
        f"util {selected['utilization_gpu']}%)."
    )
    return selected
