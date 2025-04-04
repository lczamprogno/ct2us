"""
CT2US Pipeline Implementation.

This module contains the main CT2US pipeline class that orchestrates the segmentation,
ultrasound rendering, and point cloud generation process.
"""
import os
import time
from pathlib import PosixPath as pthlib
import nibabel
import tqdm
from typing import Dict, List, Tuple, Union, Optional, Any

import numpy as np
import torch
from torch import nn, device
from torchvision import transforms
from nibabel import nifti1
from PIL import Image

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cusci
except ImportError:
    print("Error loading cupy and cusci, GPU not available?")

import scipy.ndimage
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO


class CT2USPipeline(nn.Module):
    """
    Main pipeline for CT2US that converts CT volumes to simulated ultrasound images.
    """
    def __init__(self, 
                 device_str: str = 'cuda', 
                 segmentation=None,
                 us_renderer=None,
                 pcd_sampler=None,
                 save_intermediates: bool = False,
                 intermediate_dir: str = './intermediates',
                 force_cpu: bool = False,
                 method: str = 'standard',
                 us_method: str = 'lotus_new'):
        """Initialize the CT2US pipeline.
        
        Args:
            device_str: Device to use for computation ('cuda' or 'cpu')
            segmentation: Optional segmentation component
            us_renderer: Optional ultrasound rendering component
            pcd_sampler: Optional point cloud sampling component
            save_intermediates: Whether to save intermediate results
            intermediate_dir: Directory to save intermediate results
            force_cpu: Force CPU usage even if CUDA is available
            method: Segmentation method ('standard', 'predictor', or 'mp_predictor')
            us_method: Ultrasound rendering method ('lotus_new' or 'new')
        """
        super(CT2USPipeline, self).__init__()
        
        # Set device based on force_cpu flag
        self.force_cpu = force_cpu
        if force_cpu:
            device_str = 'cpu'
            
            # Disable CUDA visibility if forcing CPU
            if torch.cuda.is_available():
                import os
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
            
        self.device = device_str
        self.save_intermediates = save_intermediates
        self.intermediate_dir = intermediate_dir
        
        # Set segmentation and US rendering methods
        self.method = method
        self.us_method = us_method
        
        # Create intermediate directory if needed
        if save_intermediates:
            import os
            os.makedirs(intermediate_dir, exist_ok=True)
        
        # Setup computational module based on device
        if device_str != 'cpu' and torch.cuda.is_available() and not force_cpu:
            try:
                self.m = cp
                self.ops = cusci
            except (ImportError, NameError):
                self.m = np
                self.ops = scipy.ndimage
        else:
            self.m = np
            self.ops = scipy.ndimage
        
        # Set up components
        self.segmentation = segmentation
        self.us_renderer = us_renderer
        self.pcd_sampler = pcd_sampler
        
        # Create default components if not provided
        if segmentation is None or us_renderer is None or pcd_sampler is None:
            self._create_default_components()
        
        # Palette data for visualization
        self.palettedata = [
            0, 0, 0, 0, 0, 0, 220, 30, 30, 170, 80, 0, 0, 170, 0, 0, 0, 0, 
            0, 175, 20, 0, 0, 0, 0, 170, 190, 0, 0, 0, 0, 0, 0, 0, 120, 230, 
            115, 65, 200, 255, 0, 150
        ]
    
    def _determine_segmentation_device(self, gpu_memory_threshold_gb=4):
        """
        Determine whether to use CPU or GPU for segmentation
        based on available GPU memory.
        
        Args:
            gpu_memory_threshold_gb: Minimum required free GPU memory in GB
            
        Returns:
            str: 'cuda' or 'cpu'
        """
        # Default to CPU for safety
        device = 'cpu'
        
        # Skip if user explicitly requested CPU
        if self.device == 'cpu':
            return 'cpu'
        
        # Check if CUDA is available
        try:
            import torch
            if torch.cuda.is_available():
                # Get available GPU memory in GB
                try:
                    # Get free memory for current GPU
                    gpu_id = torch.cuda.current_device()
                    total_mem = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)  # GB
                    allocated_mem = torch.cuda.memory_allocated(gpu_id) / (1024**3)  # GB
                    free_mem = total_mem - allocated_mem
                    
                    print(f"GPU memory status - Total: {total_mem:.2f}GB, Used: {allocated_mem:.2f}GB, Free: {free_mem:.2f}GB")
                    
                    # Use GPU if enough memory is available
                    if free_mem > gpu_memory_threshold_gb:
                        device = 'cuda'
                        print(f"Using GPU for segmentation (free memory: {free_mem:.2f}GB)")
                    else:
                        print(f"Not enough GPU memory for segmentation: {free_mem:.2f}GB available, {gpu_memory_threshold_gb:.2f}GB minimum required")
                        print("Fallback to CPU for segmentation for stability")
                except Exception as e:
                    print(f"Error checking GPU memory: {e}. Using CPU for safety.")
        except:
            print("Could not determine GPU memory. Using CPU for segmentation.")
        
        return device
    
    def _create_default_components(self):
        """Create default components if not provided."""
        # Import the component classes
        try:
            from pipeline.component_classes import (
                Config, TotalSegmentator,
                LotusUltrasoundRenderer,
                PointCloudSampler,
                PredictorSegmentator
            )
            # Try to import the MP segmentator
            try:
                from pipeline.mp_segmentator import MPPredictorSegmentator
                has_mp_segmentator = True
            except ImportError:
                has_mp_segmentator = False
        except ImportError:
            from ct2us.pipeline.component_classes import (
                Config, TotalSegmentator,
                LotusUltrasoundRenderer,
                PointCloudSampler,
                PredictorSegmentator
            )
            # Try to import the MP segmentator
            try:
                from ct2us.pipeline.mp_segmentator import MPPredictorSegmentator
                has_mp_segmentator = True
            except ImportError:
                has_mp_segmentator = False
        
        # Create configuration
        config = Config(
            save_intermediates=self.save_intermediates,
            intermediate_dir=self.intermediate_dir
        )
        
        seg_device = self._determine_segmentation_device()

        if self.method == 'predictor' or self.method == 'mp_predictor':
            try:
                from totalsegmentator.config import setup_nnunet, setup_totalseg
                setup_nnunet()
                setup_totalseg()
                try:
                    from pipeline.initialize_predictors import initialize_predictors
                except ImportError:
                    from ct2us.pipeline.initialize_predictors import initialize_predictors
                
                predictors = initialize_predictors(device=seg_device, folds=[0])
                
                if self.method == 'mp_predictor' and has_mp_segmentator:
                    # Calculate number of worker processes based on available resources
                    num_cpu_workers = max(1, os.cpu_count() // 2)  # Use half available CPU cores
                    num_gpu_workers = 1 if not self.force_cpu and torch.cuda.is_available() else 0
                    
                    # Create kwargs with all necessary parameters
                    mp_kwargs = {
                        'device': seg_device,
                        'force_cpu': self.force_cpu,
                        'num_cpu_workers': num_cpu_workers,
                        'num_gpu_workers': num_gpu_workers,
                        'save_intermediates': self.save_intermediates,
                        'intermediate_dir': self.intermediate_dir
                    }
                    
                    # Create multiprocessing segmentator
                    self.segmentation = MPPredictorSegmentator(mp_kwargs, predictors)
                    print(f"Using MultiProcessing PredictorSegmentator with {num_gpu_workers} GPU workers and {num_cpu_workers} CPU workers")
                else:
                    # Create standard predictor segmentator
                    self.segmentation = PredictorSegmentator({'device': seg_device, 'force_cpu': self.force_cpu}, predictors)
                    print(f"Using standard PredictorSegmentator on {seg_device}")
            except ImportError as e:
                print(f"Warning: Predictor method requested but TotalSegmentator not available ({e}). Using standard segmentation.")
                self.segmentation = TotalSegmentator({'device': seg_device, 'force_cpu': self.force_cpu})
        else:
            self.segmentation = TotalSegmentator({'device': seg_device, 'force_cpu': self.force_cpu})
            print(f"Note: Using {seg_device} for TotalSegmentator based on available resources")
    
        # Create US rendering component
        if self.us_renderer is None:
            if self.us_method == 'new':
                self.us_renderer = OptimizedLotusRenderer(self.device, config)
            else:  # 'lotus_new'
                self.us_renderer = LotusUltrasoundRenderer(self.device, config)
        
        # Create point cloud sampler
        if self.pcd_sampler is None:
            self.pcd_sampler = PointCloudSampler(self.device, config)
            
    
    def _save(self, name: list[str], data: list[Union[np.ndarray, torch.Tensor]], properties, dir: str = None):
        """Save intermediate result.
        """
        if dir is None:
            dir = self.intermediate_dir

            if not self.save_intermediates:
                return
            
        for i in range(len(name)):
            # Convert to numpy if needed
            d = data[i]
            n = name[i]

            if isinstance(d, torch.Tensor):
                if d.is_cuda:
                    d_np = d.detach().cpu().numpy()
                else:
                    d_np = d.detach().numpy()
            elif isinstance(d, cp.ndarray):
                d_np = d.get()
            else:
                d_np = d

            SimpleITKIO().write_seg(
                d_np.transpose(2, 1, 0), 
                n if dir==None else os.path.join(dir, f"{n}.nii.gz"),
                properties[i]
            )
               
    def forward(self,
                imgs: List[Union[nifti1.Nifti1Image, np.ndarray]],
                properties: List[Dict],
                dest_label: List[str],
                dest_us: List[str],
                step_size: int,
                save_labels: bool = True) -> Tuple[List[str], List, List, List, Dict]:
        """Run the CT2US pipeline to convert CT volumes to ultrasound images.
        
        Args:
            imgs: List of input CT volumes
            properties: List of dictionaries containing properties for each volume
            dest_label: List of destination paths for segmentation labels
            dest_us: List of destination paths for ultrasound images
            step_size: Step size for slicing the volume
            save_labels: Whether to save segmentation labels
            
        Returns:
            Tuple containing:
            - List of destination label names
            - List of ultrasound images
            - List of warped labels
            - List of viewable label images
            - Dictionary with timing information
        """
        # Initialize timing dictionary
        timing = {
            'start_time': time.time(),
            'segmentation_time': 0,
            'assembly_time': 0,
            'us_rendering_time': 0,
            'total_time': 0
        }

        # TODO: Add preprocessing function to segmentation components and run it here
        
        # Prepare base images and empty label tensors
        # if self.method == 'predictor':
        #     # GPU-accelerated path with stacked tensors
        #     bases = torch.stack([
        #         torch.as_tensor(img, dtype=torch.float32, device=self.device).squeeze() 
        #         for img in imgs
        #     ]).cuda(self.device)
            
        #     f_labels = torch.stack([
        #         torch.zeros(bases[i].shape, dtype=torch.uint8, device=self.device) 
        #         for i in range(bases.shape[0])
        #     ], axis=0).cuda(self.device)
            
        #     # Save intermediate if requested
        #     if self.save_intermediates:
        #         self._save_intermediate("initial_labels_tensor", f_labels)
        # else:
        #     # Regular path with lists

        print(f"Using {self.method} method for segmentation")

        bases = [self.m.array(img.dataobj, dtype=self.m.float32) for img in imgs]
        f_labels = [self.m.zeros(bases[idx].shape, dtype=self.m.uint8) for idx in range(len(imgs))]
    
        # Run segmentation for each task
        segmentation_start = time.time()
        segmentation_results = []
        tasks = self.segmentation.tasks()

        print(tasks)

        for idx in tqdm.tqdm(range(len(tasks)), desc="Segmenting"):
            # Only apply fast mode for 'total' task, never for tissue_types or body
            if tasks[idx] != 'total' and hasattr(self.segmentation, 'segmentation_params'):
                # Store original fast setting
                original_fast = self.segmentation.segmentation_params.get('fast', False)
                
                # If we're using fast mode globally but this isn't a 'total' task
                if original_fast:
                    # Force fast mode off for non-total tasks
                    self.segmentation.segmentation_params['fast'] = False
                    print(f"*** Disabling fast mode for '{tasks[idx]}' segmentation regardless of global setting ***")
                    result = self.segmentation.segment(imgs, properties, tasks[idx], 4)
                    
                    # Restore original fast setting
                    self.segmentation.segmentation_params['fast'] = original_fast
                else:
                    result = self.segmentation.segment(imgs, properties, tasks[idx], 4)
            else:
                # For 'total' task, use whatever settings are configured
                result = self.segmentation.segment(imgs, properties, tasks[idx], 4)
                
            segmentation_results.append(result)

        print(segmentation_results)

        if self.save_intermediates and len(segmentation_results) == 3:
            total, tissue, body = segmentation_results
            total_n, tissue_n, body_n = [], [], []
            for idx in range(len(bases)):
                total_n.append(f"total_{pthlib(dest_label[idx]).name}")
                tissue_n.append(f"tissue_{pthlib(dest_label[idx]).name}")
                body_n.append(f"body_{pthlib(dest_label[idx]).name}")

            self._save(total_n, total, properties, self.intermediate_dir)
            self._save(tissue_n, tissue, properties, self.intermediate_dir)
            self._save(body_n, body, properties, self.intermediate_dir)


        timing['segmentation_time'] = time.time() - segmentation_start
        
        # Assemble segmentations into final label maps
        assembly_start = time.time()
        for idx in tqdm.tqdm(range(len(tasks)), desc="Composing into suitable intermediate"):
            f_labels = self.segmentation.assemble(tasks[idx], segmentation_results[idx], bases, f_labels)

        print(f_labels)

        timing['assembly_time'] = time.time() - assembly_start

        # Generate ultrasound images and warped labels
        us_rendering_start = time.time()
        # Check the renderer's render method signature
        if 'render' in dir(self.us_renderer) and callable(self.us_renderer.render):
            # Try to get the parameter count to determine correct call
            import inspect
            sig = inspect.signature(self.us_renderer.render)
            
            # Use appropriate call based on parameter count
            if len(sig.parameters) == 4:  # segs, properties, dest_us, step_size
                us_imgs, warped_labels = self.us_renderer.render(
                    f_labels.copy(), properties, dest_us, step_size
                )
            else:
                # Fallback to basic call with just necessary parameters
                try:
                    us_result = self.us_renderer.render(f_labels.copy(), dest_us, step_size)
                    if isinstance(us_result, tuple) and len(us_result) >= 2:
                        us_imgs, warped_labels = us_result[0], us_result[1]
                    else:
                        us_imgs, warped_labels = us_result, []
                except Exception as e:
                    print(f"Error in US rendering: {e}")
                    # Create empty results if rendering fails
                    us_imgs = []
                    warped_labels = []
                    for i in range(len(f_labels)):
                        us_imgs.append([])
                        warped_labels.append([])
                
        else:
            print("Warning: US renderer missing render method")
            # Create empty results
            us_imgs = []
            warped_labels = []
            for i in range(len(f_labels)):
                us_imgs.append([])
                warped_labels.append([])
            
        timing['us_rendering_time'] = time.time() - us_rendering_start
    
        # Store labelmaps for point cloud generation - handle empty cases
        try:
            self.pcd_sampler.store_labelmaps(f_labels.copy())
        except Exception as e:
            print(f"Error storing labelmaps: {e}")
            # Ensure pcd_sampler has labelmaps even if empty
            self.pcd_sampler.store_labelmaps([np.zeros((100, 100, 100), dtype=np.uint8)])
        
        # Save segmentation labels if requested
        if save_labels:
            for idx in tqdm.tqdm(range(len(f_labels)), desc="Saving labels"):
                if self.m == cp:
                    SimpleITKIO().write_seg(
                        f_labels[idx].get().transpose(2, 1, 0), 
                        dest_label[idx], properties[idx]
                    )
                else:
                    SimpleITKIO().write_seg(
                        f_labels[idx].transpose(2, 1, 0), 
                        dest_label[idx], properties[idx]
                    )
                
                print(f"SAVED TO '{dest_label[idx]}'")
        
        # Generate viewable label images
        preview_labels = []
        for idx, f in enumerate(tqdm.tqdm(f_labels, desc="Generating warped labels for annotated view")):
            temp = []
            if self.m == cp:
                labels = f.get()
            else:
                labels = f
            
            for slice_idx, arr in enumerate(list(np.flip(labels.copy().transpose(2, 1, 0), 2))[::step_size]):
                img = Image.fromarray(arr)
                img.putpalette(self.palettedata * 16)
                temp.append(img)
            
            preview_labels.append(temp)
        
        # Calculate total execution time
        timing['total_time'] = time.time() - timing['start_time']
        print(f"Times: Seg={timing['segmentation_time']:.1f}s, Assembly={timing['assembly_time']:.1f}s, Render={timing['us_rendering_time']:.1f}s, Total={timing['total_time']:.1f}s")
        
        return [str(pthlib(d).name) for d in dest_label], us_imgs, warped_labels, preview_labels, timing