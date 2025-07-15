import os
import time
import torch
import subprocess
import shutil
import glob
import tempfile
import gc
import threading
from pathlib import Path
from typing import List, Optional, Union, Tuple
from cog import BasePredictor, BaseModel, Input, Path as CogPath
import yaml
from box import Box

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

def check_and_install_dependencies():
    """Check for missing dependencies and install them if needed (runtime fallback)"""
    missing_deps = []
    
    # Check for torch_cluster
    try:
        import torch_cluster
    except ImportError:
        missing_deps.append("torch_cluster")
    
    # Check for torch_scatter
    try:
        import torch_scatter
    except ImportError:
        missing_deps.append("torch_scatter")
    
    # Check for spconv
    try:
        import spconv
    except ImportError:
        missing_deps.append("spconv")
    
    if missing_deps:
        print(f"Missing dependencies detected: {missing_deps}")
        print("Attempting to install missing dependencies...")
        
        # Get torch version for PyTorch Geometric
        torch_version = torch.__version__.split("+")[0]
        cuda_version = torch.version.cuda
        
        if cuda_version:
            cuda_version = f"cu{cuda_version.replace('.', '')}"
        else:
            cuda_version = "cpu"
        
        try:
            # Install PyTorch Geometric dependencies
            if "torch_cluster" in missing_deps or "torch_scatter" in missing_deps:
                pyg_deps = [dep for dep in missing_deps if dep in ["torch_cluster", "torch_scatter"]]
                cmd = f"pip install {' '.join(pyg_deps)} -f https://data.pyg.org/whl/torch-{torch_version}+{cuda_version}.html --no-cache-dir"
                print(f"Installing PyTorch Geometric dependencies: {cmd}")
                subprocess.run(cmd, shell=True, check=True)
            
            # Install spconv
            if "spconv" in missing_deps:
                spconv_version = f"-{cuda_version}" if cuda_version != "cpu" else ""
                cmd = f"pip install spconv{spconv_version}"
                print(f"Installing spconv: {cmd}")
                subprocess.run(cmd, shell=True, check=True)
                
            print("Successfully installed missing dependencies")
            
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to install some dependencies: {e}")
            print("Continuing with available dependencies...")

# Call immediately so downstream imports see the vars
setup_environment()

# Check and install dependencies if needed
check_and_install_dependencies()

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
# Helper functions (from app.py)
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

def validate_input_file(file_path: str) -> bool:
    """Validate if the input file format is supported."""
    supported_formats = ['.obj', '.fbx', '.FBX', '.dae', '.glb', '.gltf', '.vrm']
    if not file_path or not Path(file_path).exists():
        return False
    
    file_ext = Path(file_path).suffix.lower()
    return file_ext in supported_formats

def extract_mesh_python(input_file: str, output_dir: str) -> str:
    """Extract mesh data from 3D model using Python (replaces extract.sh)"""
    from src.data.extract import extract_builtin, get_files
    
    # Create extraction parameters
    files = get_files(
        data_name="raw_data.npz",
        inputs=str(input_file),
        input_dataset_dir=None,
        output_dataset_dir=output_dir,
        force_override=True,
        warning=False,
    )
    
    if not files:
        raise RuntimeError("No files to extract")
    
    # Run the actual extraction
    timestamp = str(int(time.time()))
    extract_builtin(
        output_folder=output_dir,
        target_count=50000,
        num_runs=1,
        id=0,
        time=timestamp,
        files=files,
    )
    
    expected_npz_dir = files[0][1]
    expected_npz_file = Path(expected_npz_dir) / "raw_data.npz"
    
    if not expected_npz_file.exists():
        raise RuntimeError(f"Extraction failed: {expected_npz_file} not found")
    
    return expected_npz_dir

def run_inference_python(
    input_file: str, 
    output_file: str, 
    inference_type: str, 
    seed: int = 12345, 
    npz_dir: str = None,
    skeleton_type: str = "articulationxl"
) -> str:
    """Unified inference function for both skeleton and skin inference."""
    import lightning as L
    from src.data.datapath import Datapath
    from src.data.dataset import DatasetConfig, UniRigDatasetModule
    from src.data.transform import TransformConfig
    from src.inference.download import download
    from src.model.parse import get_model
    from src.system.parse import get_system, get_writer
    from src.tokenizer.parse import get_tokenizer
    from src.tokenizer.spec import TokenizerConfig

    # Set random seed for skeleton inference
    if inference_type == "skeleton":
        L.seed_everything(seed, workers=True)
    
    # Load configurations based on inference type
    if inference_type == "skeleton":
        task_config_path = "configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml"
        transform_config_path = "configs/transform/inference_ar_transform.yaml"
        model_config_path = "configs/model/unirig_ar_350m_1024_81920_float32.yaml"
        system_config_path = "configs/system/ar_inference_articulationxl.yaml"
        tokenizer_config_path = "configs/tokenizer/tokenizer_parts_articulationxl_256.yaml"
        data_name = "raw_data.npz"
    else:  # skin
        task_config_path = "configs/task/quick_inference_unirig_skin.yaml"
        transform_config_path = "configs/transform/inference_skin_transform.yaml"
        model_config_path = "configs/model/unirig_skin.yaml"
        system_config_path = "configs/system/skin.yaml"
        tokenizer_config_path = None
        data_name = "predict_skeleton.npz"
    
    # Load task configuration
    with open(task_config_path, 'r') as f:
        task = Box(yaml.safe_load(f))
    
    # Setup data directory and datapath
    if inference_type == "skeleton":
        if npz_dir is None:
            npz_dir = Path(output_file).parent / "npz"
        npz_dir = Path(npz_dir)
        npz_dir.mkdir(exist_ok=True)
        npz_data_dir = extract_mesh_python(input_file, npz_dir)
        datapath = Datapath(files=[npz_data_dir], cls=None)
    else:  # skin
        skeleton_work_dir = Path(input_file).parent
        all_npz_files = list(skeleton_work_dir.rglob("**/*.npz"))
        if not all_npz_files:
            raise RuntimeError(f"No NPZ files found for skin inference in {skeleton_work_dir}")
        skeleton_npz_dir = all_npz_files[0].parent
        datapath = Datapath(files=[str(skeleton_npz_dir)], cls=None)
    
    # Load common configurations
    data_config = Box(yaml.safe_load(open("configs/data/quick_inference.yaml", 'r')))
    transform_config = Box(yaml.safe_load(open(transform_config_path, 'r')))
    
    # Setup tokenizer and model
    if inference_type == "skeleton":
        tokenizer_config = TokenizerConfig.parse(config=Box(yaml.safe_load(open(tokenizer_config_path, 'r'))))
        tokenizer = get_tokenizer(config=tokenizer_config)
        model_config = Box(yaml.safe_load(open(model_config_path, 'r')))
        model = get_model(tokenizer=tokenizer, **model_config)
    else:  # skin
        tokenizer_config = None
        tokenizer = None
        model_config = Box(yaml.safe_load(open(model_config_path, 'r')))
        model = get_model(tokenizer=None, **model_config)
    
    # Setup datasets and transforms
    predict_dataset_config = DatasetConfig.parse(config=data_config.predict_dataset_config).split_by_cls()
    predict_transform_config = TransformConfig.parse(config=transform_config.predict_transform_config)
    
    # Create data module
    data = UniRigDatasetModule(
        process_fn=model._process_fn,
        predict_dataset_config=predict_dataset_config,
        predict_transform_config=predict_transform_config,
        tokenizer_config=tokenizer_config,
        debug=False,
        data_name=data_name,
        datapath=datapath,
        cls=None,
    )
    
    # Setup callbacks and writer
    callbacks = []
    writer_config = task.writer.copy()
    
    if inference_type == "skeleton":
        writer_config['npz_dir'] = str(npz_dir)
        writer_config['output_dir'] = str(Path(output_file).parent)
        writer_config['output_name'] = Path(output_file).name
        writer_config['user_mode'] = False
    else:  # skin
        writer_config['npz_dir'] = str(skeleton_npz_dir)
        writer_config['output_name'] = str(output_file)
        writer_config['user_mode'] = True
        writer_config['export_fbx'] = True
    
    callbacks.append(get_writer(**writer_config, order_config=predict_transform_config.order_config))
    
    # Get system
    system_config = Box(yaml.safe_load(open(system_config_path, 'r')))
    
    # Dynamically set skeleton type for skeleton inference
    if inference_type == "skeleton":
        system_config.generate_kwargs.assign_cls = skeleton_type
        print(f"Using skeleton type: {skeleton_type}")
    
    system = get_system(**system_config, model=model, steps_per_epoch=1)
    
    # Setup trainer
    trainer_config = task.trainer
    resume_from_checkpoint = download(task.resume_from_checkpoint)
    
    trainer = L.Trainer(callbacks=callbacks, logger=None, **trainer_config)
    
    # Run prediction
    trainer.predict(system, datamodule=data, ckpt_path=resume_from_checkpoint, return_predictions=False)
    
    # Handle output file location and validation
    if inference_type == "skeleton":
        input_name_stem = Path(input_file).stem
        actual_output_dir = Path(output_file).parent / input_name_stem
        actual_output_file = actual_output_dir / "skeleton.fbx"
        
        if not actual_output_file.exists():
            alt_files = list(Path(output_file).parent.rglob("skeleton.fbx"))
            if alt_files:
                actual_output_file = alt_files[0]
            else:
                raise RuntimeError(f"Skeleton FBX file not found. Expected at: {actual_output_file}")
        
        if actual_output_file != Path(output_file):
            shutil.copy2(actual_output_file, output_file)
    else:  # skin
        if not Path(output_file).exists():
            skin_files = list(Path(output_file).parent.rglob("*skin*.fbx"))
            if skin_files:
                actual_output_file = skin_files[0]
                shutil.copy2(actual_output_file, output_file)
            else:
                raise RuntimeError(f"Skin FBX file not found. Expected at: {output_file}")
    
    return str(output_file)

def merge_results_python(source_file: str, target_file: str, output_file: str) -> str:
    """Merge results using Python (replaces merge.sh)"""
    from src.inference.merge import transfer
    
    # Validate input paths
    if not Path(source_file).exists():
        raise ValueError(f"Source file does not exist: {source_file}")
    if not Path(target_file).exists():
        raise ValueError(f"Target file does not exist: {target_file}")
    
    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use the transfer function directly
    transfer(source=str(source_file), target=str(target_file), output=str(output_path), add_root=False)
    
    # Validate that the output file was created
    if not output_path.exists():
        raise RuntimeError(f"Merge failed: Output file not created at {output_path}")
    
    return str(output_path.resolve())

# Thread lock to prevent multiple jobs fighting for VRAM at once
_GPU_SEM = threading.Semaphore(1)

SUPPORTED_SUFFIX = {"obj", "fbx", "FBX", "dae", "glb", "gltf", "vrm"}

# -----------------------------------------------------------------------------
# Output models
# -----------------------------------------------------------------------------

class Output(BaseModel):
    skeleton_only: Optional[CogPath] = None
    skeleton_and_skin: Optional[CogPath] = None
    message: str
    processing_time: float

# -----------------------------------------------------------------------------
# Main predictor class
# -----------------------------------------------------------------------------

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Fast setup - models loaded on-demand for optimal cold start"""
        print("UniRig predictor setup completed - models will load on-demand")
        
    def predict(
        self,
        model_file: CogPath = Input(
            description="Input 3D asset (obj, fbx, glb, gltf, vrm)",
        ),
        pipeline_mode: str = Input(
            description="Pipeline mode: 'skeleton_only' for just skeleton generation, 'full' for skeleton + skinning",
            choices=["skeleton_only", "full"],
            default="skeleton_only",
        ),
        skeleton_type: str = Input(
            description="Skeleton type: 'articulationxl' for generic names, 'vroid' for descriptive names",
            choices=["articulationxl", "vroid"],
            default="articulationxl",
        ),
        seed: int = Input(
            description="Random seed for reproducible results",
            default=12345,
        ),
    ) -> Output:
        """Run UniRig pipeline with skeleton generation and optional skinning"""
        
        # Ensure model is loaded (lazy loading)
        _ensure_model_loaded()
        
        # Validate input file
        if not validate_input_file(str(model_file)):
            raise ValueError(f"Invalid or unsupported file format. Supported formats: {', '.join(SUPPORTED_SUFFIX)}")
        
        # Create working directory
        work_dir = Path(tempfile.mkdtemp())
        file_stem = Path(model_file).stem
        
        # Copy input file to working directory
        input_file = work_dir / Path(model_file).name
        shutil.copy2(model_file, input_file)
        
        start_time = time.time()
        
        try:
            with _GPU_SEM:
                print(f"Running UniRig pipeline in {pipeline_mode} mode with {skeleton_type} skeleton type...")
                
                # Step 1: Generate skeleton
                intermediate_skeleton_file = work_dir / f"{file_stem}_skeleton.fbx"
                final_skeleton_file = work_dir / f"{file_stem}_skeleton_only{input_file.suffix}"
                
                run_inference_python(
                    str(input_file), 
                    str(intermediate_skeleton_file), 
                    "skeleton", 
                    seed, 
                    skeleton_type=skeleton_type
                )
                
                merge_results_python(
                    str(intermediate_skeleton_file), 
                    str(input_file), 
                    str(final_skeleton_file)
                )
                
                skeleton_only_result = final_skeleton_file
                skeleton_and_skin_result = None
                
                # Step 2: Generate skinning if in full mode
                if pipeline_mode == "full":
                    intermediate_skin_file = work_dir / f"{file_stem}_skin.fbx"
                    final_skin_file = work_dir / f"{file_stem}_skeleton_and_skinning{input_file.suffix}"
                    
                    run_inference_python(
                        str(intermediate_skeleton_file), 
                        str(intermediate_skin_file), 
                        "skin"
                    )
                    
                    merge_results_python(
                        str(intermediate_skin_file), 
                        str(input_file), 
                        str(final_skin_file)
                    )
                    
                    skeleton_and_skin_result = final_skin_file
                
                # Clean up GPU memory
                _cleanup_gpu_memory()
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                message = f"UniRig {pipeline_mode} pipeline completed in {processing_time:.2f} seconds using {skeleton_type} skeleton type"
                print(message)
                
                return Output(
                    skeleton_only=CogPath(skeleton_only_result) if skeleton_only_result else None,
                    skeleton_and_skin=CogPath(skeleton_and_skin_result) if skeleton_and_skin_result else None,
                    message=message,
                    processing_time=processing_time
                )
                
        except Exception as e:
            _cleanup_gpu_memory()
            raise RuntimeError(f"UniRig pipeline failed: {str(e)}")
        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(work_dir, ignore_errors=True)
            except:
                pass 