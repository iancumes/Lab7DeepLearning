"""Configuración, persistencia atómica y evidencia de ejecución."""
from __future__ import annotations
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path
import numpy as np
import torch

SEED = 23236
ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
DATA = ROOT / "data"
CHECKPOINTS = ROOT / "checkpoints"
CONTROL = ["king", "france", "computer", "good", "january", "run"]

def initialize():
    for folder in [ART, DATA, CHECKPOINTS, ART/"figures", ART/"runs", ART/"evaluations"]:
        folder.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False,
                              default=lambda x: x.item() if hasattr(x, "item") else str(x)), encoding="utf-8")
    temp.replace(path)

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def hardware():
    return {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "platform": platform.platform(), "processor": platform.processor(),
            "logical_cpus": os.cpu_count(), "python": platform.python_version(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None}

def experiment_configs():
    base = dict(dim=100, window=5, negatives=5, fraction=1.0, min_count=5,
                sample=1e-4, epochs=3, initial_lr=0.025, final_lr=0.0001,
                batch_size=1024, seed=SEED, ns_exponent=0.75, dynamic_window=False)
    changes = [("base100", {}), ("dim50", {"dim":50}), ("dim300", {"dim":300}),
               ("corpus25", {"fraction":0.25}), ("corpus50", {"fraction":0.5}),
               ("window2", {"window":2}), ("negative10", {"negatives":10})]
    return [dict(base, name=name, **diff) if not diff else {**base, **diff, "name":name} for name, diff in changes]
