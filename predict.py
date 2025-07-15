from typing import List
from cog import BasePredictor, Input, Path
import subprocess
import os
import shutil
import glob
import tempfile
import gc
import threading

# -----------------------------------------------------------------------------
# 0. Environment bootstrap (before any heavy imports as per playbook)
# -----------------------------------------------------------------------------

def _setup_environment():
    """Ensure CUDA paths are in env for child libs (PyTorch, custom ops, etc.)"""
    cuda_home = "/usr/local/cuda"
    os.environ.setdefault("CUDA_HOME", cuda_home)
    os.environ.setdefault("PATH", f"{cuda_home}/bin:" + os.environ.get("PATH", ""))
    ld_lib = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ.setdefault("LD_LIBRARY_PATH", f"{cuda_home}/lib64:{ld_lib}")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.6;8.9;9.0")
    os.environ.setdefault("OMP_NUM_THREADS", "1")  # avoid OpenMP thread explosion

# Call immediately so downstream imports see the vars
_setup_environment()

# -----------------------------------------------------------------------------
# Helper: GPU memory hygiene
# -----------------------------------------------------------------------------

def _cleanup_gpu_memory():
    """Free cached GPU memory and run Python GC."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass
    gc.collect()

# Thread lock to prevent multiple skeleton jobs fighting for VRAM at once
_GPU_SEM = threading.Semaphore(1)

SUPPORTED_SUFFIX = {"obj", "fbx", "FBX", "dae", "glb", "gltf", "vrm"}

class Predictor(BasePredictor):
    """Cog predictor for UniRig skeleton generation.

    This wrapper focuses on the **skeleton prediction** stage because it
    requires fewer heavy dependencies than full skinning.  It calls the
    existing `launch/inference/generate_skeleton.sh` script that comes with
    UniRig and returns the generated `.fbx` file.
    """

    def setup(self):
        """Called once when the container starts."""
        # The checkpoints are automatically downloaded by `src.inference.download`
        # when `run.py` is executed, so no additional setup is required here.
        pass

    def predict(
        self,
        model_file: Path = Input(
            description="Input 3D asset (obj, fbx, glb, gltf, vrm)",
        ),
        seed: int = Input(
            description="Random seed to allow different skeleton variations.",
            default=12345,
        ),
    ) -> Path:
        """Run skeleton generation and return the resulting FBX file path."""

        # Validate suffix
        suffix = str(model_file).split(".")[-1]
        if suffix not in SUPPORTED_SUFFIX:
            raise ValueError(
                f"Unsupported file type `{suffix}`. Supported: {', '.join(SUPPORTED_SUFFIX)}"
            )

        # Create a temporary output directory
        out_dir = tempfile.mkdtemp()

        cmd = [
            "bash",
            "launch/inference/generate_skeleton.sh",
            "--input",
            str(model_file),
            "--output_dir",
            out_dir,
            "--seed",
            str(seed),
        ]

        # Execute the UniRig script
        with _GPU_SEM:  # ensure single skeleton generation at a time to avoid OOM
            subprocess.run(cmd, check=True)

        # Find the generated FBX file
        fbx_files: List[str] = glob.glob(os.path.join(out_dir, "**", "*.fbx"), recursive=True)
        if len(fbx_files) == 0:
            # Clean up before raising error
            shutil.rmtree(out_dir, ignore_errors=True)
            raise RuntimeError("Skeleton generation finished but no .fbx output found.")

        result_path = Path(fbx_files[0])
        _cleanup_gpu_memory()
        return result_path 