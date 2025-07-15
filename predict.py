import os
import time
import torch
import subprocess
import shutil
import glob
import tempfile
import gc
import threading
from typing import List, Optional, Union
from cog import BasePredictor, BaseModel, Input, Path

# -----------------------------------------------------------------------------
# 0. Environment bootstrap (before any heavy imports as per Hunyuan3D pattern)
# -----------------------------------------------------------------------------

def setup_environment():
    """Ensure CUDA paths are in env for child libs (PyTorch, custom ops, etc.)"""
    os.environ["CUDA_HOME"] = "/usr/local/cuda"
    os.environ["PATH"] = f"{os.environ.get('CUDA_HOME', '/usr/local/cuda')}/bin:{os.environ.get('PATH', '')}"
    os.environ["LD_LIBRARY_PATH"] = f"{os.environ.get('CUDA_HOME', '/usr/local/cuda')}/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6;8.9;9.0"
    os.environ["OMP_NUM_THREADS"] = "1"

# Call immediately so downstream imports see the vars
setup_environment()

# -----------------------------------------------------------------------------
# Global variables for lazy loading (Hunyuan3D style)
# -----------------------------------------------------------------------------

model_pipeline = None
_loading_lock = threading.Lock()

def _ensure_model_loaded():
    """Lazy load model on first use"""
    global model_pipeline
    if model_pipeline is None:
        with _loading_lock:
            if model_pipeline is None:  # Double-check pattern
                print("Loading UniRig model on-demand...")
                # For UniRig, we don't need to load a heavy model - we use subprocess calls
                # This is just a placeholder to track that setup is complete
                model_pipeline = "UniRig skeleton generation ready"
                print("UniRig model loaded successfully")
    return model_pipeline

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _cleanup_gpu_memory():
    """Free cached GPU memory and run Python GC."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()

# Thread lock to prevent multiple skeleton jobs fighting for VRAM at once
_GPU_SEM = threading.Semaphore(1)

SUPPORTED_SUFFIX = {"obj", "fbx", "FBX", "dae", "glb", "gltf", "vrm"}

# -----------------------------------------------------------------------------
# Output models
# -----------------------------------------------------------------------------

class Output(BaseModel):
    skeleton: Path
    message: str

# -----------------------------------------------------------------------------
# Main predictor class
# -----------------------------------------------------------------------------

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Fast setup - models loaded on-demand for optimal cold start"""
        print("UniRig predictor setup completed - models will load on-demand")
        
        # Any critical setup that must happen at container start
        # But avoid loading heavy models here for faster cold starts
        
    def predict(
        self,
        model_file: Path = Input(
            description="Input 3D asset (obj, fbx, glb, gltf, vrm)",
        ),
        seed: int = Input(
            description="Random seed to allow different skeleton variations",
            default=12345,
        ),
    ) -> Output:
        """Run UniRig skeleton generation"""
        
        # Ensure model is loaded (lazy loading)
        _ensure_model_loaded()
        
        # Validate suffix
        suffix = str(model_file).split(".")[-1]
        if suffix not in SUPPORTED_SUFFIX:
            raise ValueError(
                f"Unsupported file type `{suffix}`. Supported: {', '.join(SUPPORTED_SUFFIX)}"
            )

        # Create a temporary output directory
        out_dir = tempfile.mkdtemp()
        
        try:
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

            # Execute the UniRig script with GPU semaphore
            with _GPU_SEM:
                print(f"Running UniRig skeleton generation with seed {seed}...")
                start_time = time.time()
                subprocess.run(cmd, check=True)
                end_time = time.time()
                
            # Find the generated FBX file
            fbx_files: List[str] = glob.glob(os.path.join(out_dir, "**", "*.fbx"), recursive=True)
            if len(fbx_files) == 0:
                raise RuntimeError("Skeleton generation finished but no .fbx output found.")

            result_path = Path(fbx_files[0])
            
            # Clean up GPU memory
            _cleanup_gpu_memory()
            
            processing_time = end_time - start_time
            message = f"Skeleton generation completed in {processing_time:.2f} seconds"
            print(message)
            
            return Output(
                skeleton=result_path,
                message=message
            )
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"UniRig skeleton generation failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during skeleton generation: {e}")
        finally:
            # Clean up temporary directory if it still exists
            if os.path.exists(out_dir):
                try:
                    shutil.rmtree(out_dir, ignore_errors=True)
                except:
                    pass
            _cleanup_gpu_memory() 