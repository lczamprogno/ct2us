"""
Component classes for CT2US pipeline.

This module contains modular components for segmentation, ultrasound rendering,
and point cloud sampling, which can be plugged into the CT2US pipeline.
"""

from abc import ABC, abstractmethod, abstractproperty
import os
from pathlib import PosixPath as pthlib
import tqdm
from typing import Dict, List, Tuple, Union, Any

import numpy as np
import torch
from torch import device
from torchvision import transforms
from nibabel import nifti1

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cusci
except ImportError:
    print("Error loading cupy and cusci, GPU not available?")
    # We'll use PyTorch for morphological operations anyway

import scipy.ndimage

from pipeline.ultrasound_rendering import UltrasoundRendering
from pipeline.base_component import BaseComponent

from pipeline.torch_morphology import binary_closing, binary_dilation, binary_fill_holes

import trimesh as tri

class Config:
    """Configuration container for pipeline components.
    
    This is a simple configuration class that holds parameters for pipeline components.
    """
    
    def __init__(self, **kwargs):
        """Initialize configuration with any kwargs that will become attributes."""
        # Set default values
        self.save_intermediates = False
        self.intermediate_dir = './intermediates'
        self.debug = False
        
        # Initialize config dictionary
        self.config = {}
        
        # Label dictionary for tissues
        self.l_dict = {
            2: 'lung',
            3: 'fat', 
            4: 'vessel', 
            6: 'kidney', 
            8: 'muscle', 
            11: 'liver', 
            12: 'soft tissue', 
            13: 'bone'
        }
        
        # Processing parameters
        self.config['binary_closing_iterations'] = 3
        self.config['binary_dilation_iterations'] = 2
        self.config['resize_size'] = (380, 380)
        self.config['crop_size'] = 256
        
        # Default point palette
        self.config['pointpalette'] = [torch.tensor([
            [0, 0, 0, 255],
            [0, 0, 0, 255],
            [220, 30, 30, 255],
            [170, 80, 0, 31],
            [0, 170, 0, 255],
            [0, 0, 0, 255],
            [0, 175, 20, 255],
            [0, 0, 0, 255],
            [0, 170, 190, 255],
            [0, 0, 0, 255],
            [0, 0, 0, 255],
            [0, 120, 230, 255],
            [115, 65, 200, 31],
            [255, 0, 150, 255]
        ]),
        [0, 0, 100000, 100000, 100000, 0, 100000, 0, 100000, 0, 0, 100000, 400000, 100000]]
        
        # Update with any provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class SegmentationMethod(BaseComponent):
    """Base class for segmentation methods in the CT2US pipeline."""

    def __init__(self, kwargs=None):
        # Ensure kwargs is a dictionary
        if kwargs is None:
            kwargs = {}
        elif isinstance(kwargs, str):
            # Handle case where a string is passed instead of a dict
            device_str = kwargs
            kwargs = {'device': device_str}
            
        super().__init__(kwargs.get('device', 'cpu'))

        self.config['use_roi'] = kwargs.get('use_roi', False)
        self.config['resamp_thr'] = kwargs.get('resamp_thr', 4)
        
        # Morphological operation parameters
        self.config['binary_dilation_iterations'] = kwargs.get('binary_dilation_iterations', 1)
        self.config['binary_erosion_iterations'] = kwargs.get('binary_erosion_iterations', 3)
        
        # Density thresholds for segmentation
        self.config['density_min'] = kwargs.get('density_min', -200)
        self.config['density_max'] = kwargs.get('density_max', 250)
        
        # Blob size thresholds for filtering
        self.config['blob_size_min'] = kwargs.get('blob_size_min', 10)
        self.config['blob_size_max'] = kwargs.get('blob_size_max', 30)
        
        # 3D kernel size for operations
        self.config['kernel_size'] = kwargs.get('kernel_size', 3)

        # Label mappings - default values from notebook
        self.config['name2label'] = kwargs.get('name2label', {
            "total": {
                "2": ["lung"],
                "4": ["aorta", "artery", "atrial", "iliac", "vein", "vena"],
                "6": ["kidney"],
                "8": ["bowel", "colon", "esophagus", "gallbladder", "heart", "stomach", 
                      "trunk", "autochlon", "iliopsoas", "gluteus"],
                "11": ["liver"],
                "12": ["adrenal_gland", "duodenum", "pancreas", "spleen"],
                "13": ["clavicula", "humerus", "rib_", "vertebrae_", "sacrum", "scapula", 
                       "sternum", "femur", "hip", "fibula", "tibia", "radius", "ulna", 
                       "carpal", "tarsal", "patella"]
            },
            "body": {
                "bg": 9,
                "skin": 12,
                "fat": 3,
                "muscle": 8
            }
        })
        
        
        # Total label map lookup from ids to class value
        self.config['total_lmap'] = {"0": 0, "1": 12, "2": 6, "3": 6, "4": 8, "5": 11, "6": 8, "7": 12, 
                          "8": 12, "9": 12, "10": 2, "11": 2, "12": 2, "13": 2, "14": 2, 
                          "15": 8, "16": 0, "17": 0, "18": 8, "19": 12, "20": 8, "21": 0, 
                          "22": 0, "23": 6, "24": 6, "25": 13, "26": 13, "27": 13, "28": 13, 
                          "29": 13, "30": 13, "31": 13, "32": 13, "33": 13, "34": 13, "35": 13, 
                          "36": 13, "37": 13, "38": 13, "39": 13, "40": 13, "41": 13, "42": 13, 
                          "43": 13, "44": 13, "45": 13, "46": 13, "47": 13, "48": 13, "49": 13, 
                          "50": 13, "51": 8, "52": 4, "53": 4, "54": 8, "55": 4, "56": 4, 
                          "57": 4, "58": 4, "59": 4, "60": 4, "61": 4, "62": 4, "63": 4, 
                          "64": 4, "65": 4, "66": 4, "67": 4, "68": 4, "69": 13, "70": 13, 
                          "71": 13, "72": 13, "73": 13, "74": 13, "75": 0, "76": 0, "77": 0, 
                          "78": 0, "79": 0, "80": 0, "81": 0, "82": 0, "83": 0, "84": 0, 
                          "85": 0, "86": 0, "87": 0, "88": 0, "89": 0, "90": 0, "91": 0, 
                          "92": 13, "93": 13, "94": 13, "95": 13, "96": 13, "97": 13, "98": 13, 
                          "99": 13, "100": 13, "101": 13, "102": 13, "103": 13, "104": 13, 
                          "105": 13, "106": 13, "107": 13, "108": 13, "109": 13, "110": 13, 
                          "111": 13, "112": 13, "113": 13, "114": 13, "115": 13, "116": 13, 
                          "117": 0}
        
        # Create tmap from total_lmap
        self.config['tmap'] = self._dict_2_map(self.config['total_lmap'])
    
    def _dict_2_map(self, d: Dict[str, int]) -> List[List[int]]:
        """Convert a dictionary to a map structure for label lookup.
        
        Args:
            d: Dictionary mapping string keys to integer values
            
        Returns:
            A map structure for label lookup
        """
        result_map = [[] for _ in range(15)]  # Assuming max 15 different labels
        
        for k, v in d.items():
            int_k = int(k)
            result_map[v].append(int_k)
            
        return result_map

    @abstractmethod
    def segment(self, 
                imgs: List[Union[nifti1.Nifti1Image, np.ndarray]],
                properties: List[Dict],
                task: str,
                resamp_thr: int) -> List[np.ndarray]:
        """Segment the input images for the given task.
        
        Args:
            imgs: List of input images to segment
            properties: List of dictionaries containing image properties
            task: The segmentation task to perform ('total', 'tissue_types', 'body')
            resamp_thr: Resampling threshold
            
        Returns:
            List of segmentation results
        """
        pass

    @abstractmethod
    def assemble(self,
                task: str,
                segs: List[np.ndarray],
                bases: List[np.ndarray],
                prev: List[np.ndarray]) -> List[np.ndarray]:
        """Assemble segmentation results into a unified segmentation map.
        
        Args:
            task: The task that was performed ('total', 'tissue_types', 'body')
            segs: Segmentation results
            bases: Base images
            prev: Previous segmentation results to update
            
        Returns:
            Updated segmentation results
        """
        pass

    def name():
        return ""
    
    @property
    @abstractmethod
    def tasks(self):
        """Segmentation tasks to iterate through"""
        pass


class TotalSegmentator(SegmentationMethod):
    """Segmentation method using TotalSegmentator."""
    
    def __init__(self, kwargs=None):
        """Initialize TotalSegmentator method.
        
        Args:
            kwargs: Dictionary containing configuration parameters including:
                device: The device to use for computation ('cuda' or 'cpu')
                Other configuration parameters
        """
        # Ensure kwargs is a dictionary
        if kwargs is None:
            kwargs = {}
        elif isinstance(kwargs, str):
            # Handle case where a string is passed instead of a dict
            device_str = kwargs
            kwargs = {'device': device_str}
            
        # Check CUDA availability and set device correctly
        import torch
        if 'device' in kwargs and kwargs['device'] == 'cuda' and not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.")
            kwargs['device'] = 'cpu'
        elif 'device' not in kwargs and torch.cuda.is_available():
            print("CUDA detected and will be used.")
            kwargs['device'] = 'cuda'
        
        super().__init__(kwargs)
        
        # Ensure license key is set if provided
        if 'license_key' in kwargs:
            import os
            os.environ['TS_LICENSE_KEY'] = kwargs['license_key']
            print(f"Set TotalSegmentator license key from kwargs")
        else:
            # Check if license key is already defined globally
            try:
                import __main__
                if hasattr(__main__, 'license') and __main__.license:
                    import os
                    os.environ['TS_LICENSE_KEY'] = __main__.license
                    print(f"Set TotalSegmentator license key from global variable")
            except Exception as e:
                print(f"Note: No license key found in kwargs or globally: {e}")
        
        # Import TotalSegmentator after setting license key
        import totalsegmentator.python_api as ts
        self.ts = ts

        # Configure segmentation parameters based on task
        self.segmentation_params = {
            "force_split": True,  # Split large images if needed
            "fast": kwargs.get('fast', False),  # Allow fast mode to be configured
            "verbose": kwargs.get('verbose', True)  # Add verbose output for debugging
        }
        
    
    def segment(self, 
                imgs: List[Union[nifti1.Nifti1Image, np.ndarray]],
                properties: List[Dict],
                task: str,
                resamp_thr: int) -> List[np.ndarray]:
        """Segment the input images using TotalSegmentator.
        
        Args:
            imgs: List of input images to segment
            properties: List of dictionaries containing image properties
            task: The segmentation task to perform ('total', 'tissue_types', 'body')
            resamp_thr: Resampling threshold
            
        Returns:
            List of segmentation results
        """
        # Store the mapped task for use in segmentation
        ret = []
        
        # Set license key from environment if available
        import os
        import torch
        import gc
        
        # Double-check CUDA availability and set device properly
        cuda_available = torch.cuda.is_available()
        print(f"CUDA detected: {cuda_available}")
        
        # Set environment variable for TotalSegmentator to use GPU/CPU
        if cuda_available and str(self.device).startswith('cuda'):
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''   # Disable GPU
        
        # Ensure license key is set
        try:
            from __main__ import license
            if license and not os.environ.get('TS_LICENSE_KEY'):
                os.environ['TS_LICENSE_KEY'] = license
                print(f"Set TotalSegmentator license key from global variable")
        except Exception as e:
            print(f"Note: No license key found globally: {e}")
        
        # Clear CUDA cache before starting to ensure maximum available memory
        try:
            if cuda_available:
                torch.cuda.empty_cache()
                gc.collect()
                print("Cleared CUDA cache")
        except Exception as e:
            print(f"Error clearing CUDA cache: {e}")
        
        # Process each image
        for img in imgs:
            print(f"Processing image with TotalSegmentator (on device: {self.device}) for task: {task}...")
            try:
                # Create segmentation with specified task
                segmentation = self.ts.totalsegmentator(
                    input=img,
                    task=task,
                    nr_thr_resamp=resamp_thr,
                    fast=self.segmentation_params.get("fast", False)
                )

                print(f"Segmentation completed successfully")
                
                # Convert the segmentation result to numpy array
                if hasattr(segmentation, 'dataobj'):
                    seg_array = np.asarray(segmentation.dataobj, dtype=np.uint8)
                else:
                    seg_array = np.asarray(segmentation, dtype=np.uint8)
                
                print(f"Converted to array with shape: {seg_array.shape}")
                
                # Add result to return list
                ret.append(seg_array)
                print(f"Completed segmentation for task: {task}")
                
            except Exception as e:
                print(f"Error in segmentation for task {task}: {e}")
                # Return empty segmentation of the right shape if there's an error
                if len(ret) > 0:
                    # Use same shape as previous successful segmentation
                    ret.append(np.zeros_like(ret[-1]))
                else:
                    # Create empty segmentation with a default shape
                    img_shape = img.shape if hasattr(img, 'shape') else (256, 256, 256)
                    ret.append(np.zeros(img_shape, dtype=np.uint8))
                print(f"Created empty segmentation due to error")
            
            # Clear memory after each image
            try:
                gc.collect()
                if cuda_available:
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error clearing memory: {e}")
            
        return ret
    
    def assemble(self,
                task: str,
                segs: List[np.ndarray],
                bases: List[np.ndarray],
                prev: List[np.ndarray]) -> List[np.ndarray]:
        """Assemble segmentation results into a unified segmentation map.
        
        Args:
            task: The task that was performed ('total', 'tissue_types', 'body')
            segs: Segmentation results
            bases: Base images
            prev: Previous segmentation results to update
            
        Returns:
            Updated segmentation results
        """
        print(f"ASSEMBLY STARTED: {task}")
        
        # Verify we have segmentation results
        if not segs:
            print(f"Warning: No segmentation results to assemble for task '{task}'")
            return prev
        
        # Print info about segmentation results
        print(f"Number of segmentation results: {len(segs)}")
        for i, seg in enumerate(segs):
            if seg is not None:
                print(f"Seg {i} shape: {seg.shape}, dtype: {seg.dtype}, unique values: {np.unique(seg)[:10]}")
            else:
                print(f"Seg {i} is None")
        
        # Simplified implementation directly from the working code example
        if task == 'total':
            for j in range(len(segs)):
                if segs[j] is None or segs[j].size == 0:
                    print(f"Skipping empty segmentation result at index {j}")
                    continue
                
                # Process each label group according to total_lmap
                for i in range(len(self.config['tmap'])):
                    if len(self.config['tmap'][i]) > 0:  # if there are any keys for this value
                        try:
                            # This exact line is from the working code
                            a = self.m.where(self.m.isin(self.m.asarray(segs[j], dtype=self.m.uint8), 
                                                       self.m.array(self.config['tmap'][i])), 
                                           self.m.uint8(i), self.m.uint8(0))
                            prev[j] += a
                        except Exception as e:
                            print(f"Error processing class {i}: {e}")
        
        elif task == 'tissue_types':
            for j in range(len(segs)):
                if segs[j] is None or segs[j].size == 0:
                    print(f"Skipping empty segmentation result at index {j}")
                    continue
                
                # This matches the working code
                t = self.m.asarray(segs[j])
                prev[j][t == 1] = self.m.uint8(self.config['name2label']["body"]["fat"])
                prev[j][t == 2] = self.m.uint8(self.config['name2label']["body"]["fat"])
        
        elif task == 'body':
            for j in range(len(segs)):
                if segs[j] is None or segs[j].size == 0:
                    print(f"Skipping empty segmentation result at index {j}")
                    continue
                
                try:
                    # Direct implementation from the working code
                    t = self.m.asarray(segs[j])
                    
                    # Values directly from the working code
                    body = self.ops.binary_dilation(t == 1, iterations=1).astype(self.m.uint8)
                    body_inner = self.ops.binary_erosion(t, iterations=3, brute_force=True).astype(self.m.uint8)
                    skin = body - body_inner
                    
                    # Range values directly from the working code
                    density_mask = (bases[j] > -200) & (bases[j] < 250)
                    skin[~density_mask] = 0
                    
                    # Process connected components to remove small blobs
                    mask, _ = self.ops.label(skin)
                    counts = self.m.bincount(mask.flatten())
                    
                    # Values directly from the working code
                    if len(counts) > 1:
                        remove = self.m.where((counts <= 10) | (counts > 30), True, False)
                        remove_idx = self.m.nonzero(remove)[0]
                        mask[self.m.isin(self.m.array(mask), remove_idx)] = 0
                        mask[mask > 0] = 1
                    
                    # Match working code using a 2x2x2 kernel
                    dilation_kernel = self.m.ones(shape=(2, 2, 2))
                    skin = self.m.where(self.ops.binary_dilation(skin == 1, structure=dilation_kernel), 
                                       self.m.uint8(1), self.m.uint8(0))
                    
                    prev[j][skin == 1] = self.m.uint8(self.config['name2label']["body"]["skin"])
                    
                    tmp = prev[j].copy()
                    prev[j][tmp == 0] = self.m.uint8(self.config['name2label']["body"]["bg"])
                except Exception as e:
                    print(f"Error processing body for segmentation {j}: {e}")
                    
        print("ASSEMBLY COMPLETED")
        return prev
    
    def name():
        return "TotalSegmentator"
    
    def tasks(self):
        return ['total', 'tissue_types', 'body']

class TotalSegmentatorFast(TotalSegmentator):
    def __init__(self, kwargs=None):
        # Ensure kwargs is a dictionary
        if kwargs is None:
            kwargs = {}
        elif isinstance(kwargs, str):
            # Handle case where a string is passed instead of a dict
            device_str = kwargs
            kwargs = {'device': device_str}
        
        # Set fast mode in kwargs to ensure it's passed to super()
        kwargs['fast'] = True
        
        # Call parent initializer with updated kwargs
        super().__init__(kwargs)
        
        # Double-check fast mode is set
        self.segmentation_params["fast"] = True
        print("Using TotalSegmentator in fast mode (3mm)")

    def name():
        return "TotalSegmentator 3mm"

class PredictorSegmentator(SegmentationMethod):
    """Segmentation method using neural network predictors."""
    
    def __init__(self, kwargs=None, predictors=None):
        """Initialize PredictorSegmentator method.
        
        Args:
            kwargs: Dictionary containing configuration parameters including:
                device: The device to use for computation ('cuda' or 'cpu')
                Other configuration parameters
            predictors: Optional dictionary of pre-initialized predictors
        """
        # Ensure kwargs is a dictionary
        if kwargs is None:
            kwargs = {}
        elif isinstance(kwargs, str):
            # Handle case where a string is passed instead of a dict
            device_str = kwargs
            kwargs = {'device': device_str}
            
        super().__init__(kwargs)

        # Extract predictors from kwargs if provided there
        if predictors is None and 'predictors' in kwargs:
            predictors = kwargs.get('predictors')
            
        self.predictors = predictors

        if predictors:
            self.predictor_keys = predictors.keys()
    
    def segment(self, 
                imgs: List[Union[nifti1.Nifti1Image, np.ndarray]],
                properties: List[Dict],
                task: str,
                resamp_thr: int) -> List[np.ndarray]:
        """Segment the input images using neural network predictors.
        
        Args:
            imgs: List of input images to segment
            properties: List of dictionaries containing image properties
            task: The segmentation task to perform (should be a predictor key)
            resamp_thr: Resampling threshold
            
        Returns:
            List of segmentation results
        """
        return self.predictors[task].predict_from_list_of_npy_arrays(
            imgs, None, properties, None, 2, save_probabilities=False,
            num_processes_segmentation_export=resamp_thr
        )
    
    def assemble(self,
                task: str,
                segs: List[np.ndarray],
                bases: List[np.ndarray],
                prev: List[np.ndarray]) -> List[np.ndarray]:
        """Assemble segmentation results into a unified segmentation map.
        
        Args:
            task: The task that was performed ('total', 'tissue_types', 'body')
            segs: Segmentation results
            bases: Base images
            prev: Previous segmentation results to update
            
        Returns:
            Updated segmentation results
        """
        print(f"ASSEMBLY STARTED: {task}")
        
        # Morphological operation parameters
        binary_dilation_iterations = self.config['binary_dilation_iterations'] 
        binary_erosion_iterations = self.config['binary_erosion_iterations']
        
        # Density thresholds for segmentation
        density_min = self.config['density_min'] 
        density_max = self.config['density_max'] 
        
        # Blob size thresholds for filtering
        blob_size_min = self.config['blob_size_min'] 
        blob_size_max = self.config['blob_size_max'] 
        
        # 3D kernel size for operations
        kernel_size = self.config['kernel_size'] 

        if task == 'total':
            for j in range(len(segs)):
                for i in range(len(self.config['tmap'])):
                    if len(self.config['tmap'][i]) > 0:  # if there are any keys for this value
                        a = self.m.where(self.m.isin(self.m.asarray(segs[j], dtype=self.m.uint8), 
                                                   self.m.array(self.config['tmap'][i])), 
                                       self.m.uint8(i), self.m.uint8(0))
                        prev[j] += a
        
        elif task == 'tissue_types':
            for j in range(len(segs)):
                t = self.m.asarray(segs[j])
                prev[j][t == 1] = self.m.uint8(self.config['name2label']["body"]["fat"])
                prev[j][t == 2] = self.m.uint8(self.config['name2label']["body"]["fat"])
        
        elif task == 'body':
            for j in range(len(segs)):
                t = self.m.asarray(segs[j])
                
                body = self.ops.binary_dilation(t == 1, iterations=binary_dilation_iterations).astype(self.m.uint8)
                body_inner = self.ops.binary_erosion(t, iterations=binary_erosion_iterations, brute_force=True).astype(self.m.uint8)
                skin = body - body_inner
                
                # Segment by density
                density_mask = (bases[j] > density_min) & (bases[j] < density_max)
                skin[~density_mask] = 0
                
                # Process connected components to remove small blobs
                mask, _ = self.ops.label(skin)
                counts = self.m.bincount(mask.flatten())
                
                if len(counts) > 1:
                    remove = self.m.where((counts <= blob_size_min) | (counts > blob_size_max), True, False)
                    remove_idx = self.m.nonzero(remove)[0]
                    mask[self.m.isin(self.m.array(mask), remove_idx)] = 0
                    mask[mask > 0] = 1
                
                shape = (kernel_size, kernel_size, kernel_size)
                dilation_kernel = self.m.ones(shape=shape)
                skin = self.m.where(self.ops.binary_dilation(skin == 1, structure=dilation_kernel), 
                                   self.m.uint8(1), self.m.uint8(0))
                
                prev[j][skin == 1] = self.m.uint8(self.config['name2label']["body"]["skin"])
                
                tmp = prev[j].copy()
                prev[j][tmp == 0] = self.m.uint8(self.config['name2label']["body"]["bg"])
                        
        print("ASSEMBLY COMPLETED")
        return prev
    
    def name():
        return "Predictor"
    
    def tasks(self):
        return self.predictor_keys

# TODO: Add more config options
class UltrasoundRenderingMethod(BaseComponent):
    """Base class for ultrasound rendering methods in the CT2US pipeline."""
    
    def __init__(self, kwargs=None):
        """Initialize ultrasound rendering method.
        
        Args:
            kwargs: Dictionary containing configuration parameters including:
                device: The device to use for computation ('cuda' or 'cpu')
                Other configuration parameters
        """
        # Ensure kwargs is a dictionary
        if kwargs is None:
            kwargs = {}
        elif isinstance(kwargs, str):
            # Handle case where a string is passed instead of a dict
            device_str = kwargs
            kwargs = {'device': device_str}
            
        super().__init__(kwargs.get('device', torch.device('cpu')))

        # Morphological operation parameters
        self.config['binary_dilation_iterations'] = kwargs.get('binary_dilation_iterations', 1)
        self.config['binary_erosion_iterations'] = kwargs.get('binary_erosion_iterations', 3)
        
        # Density thresholds for segmentation
        self.config['density_min'] = kwargs.get('density_min', -200)
        self.config['density_max'] = kwargs.get('density_max', 250)
        
        # Blob size thresholds for filtering
        self.config['blob_size_min'] = kwargs.get('blob_size_min', 10)
        self.config['blob_size_max'] = kwargs.get('blob_size_max', 30)
        
        # 3D kernel size for operations
        self.config['hparams'] = kwargs.get('hparams', 
                                                {
                                                    'debug': False,
                                                    'device': self.device     
                                                }
                                            )
        
        self.l_dict = kwargs.get('l_dict', 
                                {
                                    2:'lung',
                                    3:'fat',
                                    4:'vessel',
                                    6:'kidney',
                                    8:'muscle',
                                    11:'liver',
                                    12:'soft tissue',
                                    13:'bone'    
                                })
        
    @abstractmethod
    def render(self, 
               segs: List[np.ndarray],
               properties: Dict,
               dest_us: List[str],
               step_size: int) -> Tuple[List, List, List]:
        """Render ultrasound images from segmentations.
        
        Args:
            segs: List of segmentation results
            properties: Properties of the input images
            dest_us: Destination paths for ultrasound images
            step_size: Step size for slicing
            
        Returns:
            Tuple containing:
            - List of rendered ultrasound images
            - List of warped label maps
            - List of point cloud data
        """
        pass


class LotusUltrasoundRenderer(UltrasoundRenderingMethod):
    """Ultrasound rendering method using the LOTUS model."""
    
    def __init__(self, kwargs=None):
        """Initialize LOTUS ultrasound renderer.
        
        Args:
            kwargs: Dictionary containing configuration parameters including:
                device: The device to use for computation ('cuda' or 'cpu')
                Other configuration parameters
        """
        # Ensure kwargs is a dictionary
        if kwargs is None:
            kwargs = {}
        elif isinstance(kwargs, str):
            # Handle case where a string is passed instead of a dict
            device_str = kwargs
            kwargs = {'device': device_str}
            
        super().__init__(kwargs)
        
        # Import UltrasoundRendering class
        from pipeline.ultrasound_rendering import UltrasoundRendering
        self.ultrasound_rendering = UltrasoundRendering(self.config.get('hparams', {}), default_param=True).to(self.device)

        resize_size = kwargs.get('resize_size', [380, 380])
        crop_size = kwargs.get('crop_size', 256)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize_size, transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(crop_size),
        ])
    
    def render(self, 
               segs: List[np.ndarray],
               properties: Dict,
               dest_us: List[str],
               step_size: int) -> Tuple[List, List, List]:
        """Render ultrasound images using LOTUS model.
        
        Args:
            segs: List of segmentation results
            properties: Properties of the input images
            dest_us: Destination paths for ultrasound images
            step_size: Step size for slicing
            
        Returns:
            Tuple containing:
            - List of rendered ultrasound images
            - List of warped label maps
            - List of point cloud data (None for this renderer)
        """
        print("US SIMULATION STARTED")

        # Use UltrasoundRendering from pipeline like in the working code
        from pipeline.ultrasound_rendering import UltrasoundRendering
        
        hparams = {
            'debug': False,
            'device': self.device
        }

        us_r = UltrasoundRendering(params=hparams, default_param=True).to(self.device)

        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize([380, 380], transforms.InterpolationMode.NEAREST),
                    transforms.CenterCrop((256)),
                ])

        us_imgs = []
        warped_labels = []

        for i in tqdm.tqdm(range(len(segs)), desc="Rendering"):
            warped = []
            us_slices = []
            
            # Get labelmap from GPU if needed
            if self.m == cp:
                labelmap = segs[i].get()
            else:
                labelmap = segs[i]

            # Create destination directory
            dest = pthlib(dest_us[i]).joinpath("slice_")
            os.makedirs(dest.parent, exist_ok=True)

            # Set default device for operations
            torch.set_default_device(self.device)

            # Process each slice
            for slice_idx in tqdm.tqdm(range(0, labelmap.shape[2], step_size), desc="US slice rendering"):
                # Prepare the slice
                slice_data = labelmap[:, :, slice_idx].astype('int64')
                labelmap_slice = transform(slice_data).squeeze().to(self.device)

                # Generate ultrasound image
                us_slice = us_r(labelmap_slice)
                
                # Store result as numpy array
                us_slices.append(us_slice.cpu().numpy())
                
                # Save image to disk
                us_image_pil = transforms.ToPILImage()(us_slice.cpu().squeeze())
                us_image_pil.save(f"{dest}_{slice_idx}.png")

                # Warp slice to match US and create individual label maps
                temp = us_r.warp_img(labelmap_slice)
                
                # Process the warped image - this matches the working code
                a = self.m.fliplr(self.m.asarray(temp.detach().clone()).transpose(1, 0))

                # Create warped label maps for each tissue type - matches working code
                warped_slice_labels = []
                for tag in [2, 3, 4, 6, 8, 11, 12, 13]:
                    try:
                        # Use full morphological operations as in the working code
                        if self.m == cp:
                            # Use CuPy operations if available
                            mask = (a == tag)
            
                            closed = self.ops.binary_closing(mask, iterations=3, brute_force=True)
                            dilated = self.ops.binary_dilation(closed, iterations=2, brute_force=True)
                            filled = self.ops.binary_fill_holes(dilated)
            
                            tag_mask = np.asarray(filled.get())
                        else:
                            # Use NumPy/SciPy operations otherwise
                            mask = (a == tag)
                            closed = scipy.ndimage.binary_closing(mask, iterations=3)
                            dilated = scipy.ndimage.binary_dilation(closed, iterations=2)
                            filled = scipy.ndimage.binary_fill_holes(dilated)
                            tag_mask = filled
                        
                            # Add to warped labels

                        warped_slice_labels.append((tag_mask, self.l_dict[tag]))

                    except Exception as e:
                        print(f"Error processing tag {tag}: {e}")
                        # Create empty mask as fallback
                        if self.m == cp:
                            tag_mask = np.zeros_like(a.get(), dtype=np.uint8)
                        else:
                            tag_mask = np.zeros_like(a, dtype=np.uint8)
                        warped_slice_labels.append((tag_mask, self.l_dict[tag]))
                

                warped.append(warped_slice_labels)

                # Clean up
                del temp, a

            # Add results to output lists
            us_imgs.append(us_slices)
            warped_labels.append(warped)
            
        print("US SIMULATION COMPLETED")

        return us_imgs, warped_labels

    def name():
        return "LOTUS"

""""""
class OptimizedLotusRenderer(UltrasoundRenderingMethod):
    """Optimized version of the LOTUS renderer that processes batches of slices."""
    
    def __init__(self, kwargs=None):
        """Initialize optimized LOTUS renderer.
        
        Args:
            kwargs: Dictionary containing configuration parameters including:
                device: The device to use for computation ('cuda' or 'cpu')
                Other configuration parameters
        """
        # Ensure kwargs is a dictionary
        if kwargs is None:
            kwargs = {}
        elif isinstance(kwargs, str):
            # Handle case where a string is passed instead of a dict
            device_str = kwargs
            kwargs = {'device': device_str}
            
        super().__init__(kwargs)

        raise NotImplementedError("OptimizedLotusRenderer is not yet implemented.")
        
        # Import UltrasoundRendering class
        try:
            from pipeline.ultrasound_rendering import UltrasoundRendering
            self.ultrasound_rendering = UltrasoundRendering(self.config.get('hparams', {}), default_param=True).to(self.device)
        except ImportError as e:
            # Fallback to direct import from notebook (for backwards compatibility)
            try:
                from CT2US import UltrasoundRendering
                self.ultrasound_rendering = UltrasoundRendering(self.config.get('hparams', {}), default_param=True).to(self.device)
            except ImportError:
                raise ImportError(f"Could not import UltrasoundRendering class. Original error: {e}")
        
        # Set defaults that can be overridden by config
        self.resize_size = 380
        self.crop_size = 256
        
        # Get config values if provided
        if config:
            if hasattr(config, 'resize_size'):
                if isinstance(config.resize_size, tuple) and len(config.resize_size) == 2:
                    self.resize_size = config.resize_size[0]  # Just use the first dimension
                else:
                    self.resize_size = config.resize_size
            if hasattr(config, 'crop_size'):
                self.crop_size = config.crop_size
    
    def render(self, 
               segs: List[np.ndarray],
               properties: Dict,
               dest_us: List[str],
               step_size: int) -> Tuple[List, List, List]:
        """Render ultrasound images using optimized batch processing.
        
        Args:
            segs: List of segmentation results
            properties: Properties of the input images
            dest_us: Destination paths for ultrasound images
            step_size: Step size for slicing
            
        Returns:
            Tuple containing:
            - List of rendered ultrasound images
            - List of warped label maps
            - List of point cloud data
        """
        raise NotImplementedError("OptimizedLotusRenderer is not yet implemented.")

        print("US SIMULATION STARTED")
        us_imgs = []
        warped_results = []  # Store warped results for each volume
        pcd_data = []
        
        # For better slice selection support
        from pipeline.torch_morphology import binary_closing, binary_dilation, binary_fill_holes
        
        for i, (seg, dest_path) in enumerate(zip(tqdm.tqdm(segs, desc="Rendering volumes"), dest_us)):
            # Handle CPU/GPU differences
            if hasattr(seg, 'get'):  # CuPy array
                labelmap = seg.get()
            else:
                labelmap = seg
                
            # Create output directory
            dest = pthlib(dest_path).joinpath("slice_")
            os.makedirs(dest.parent, exist_ok=True)
            
            # Containers for results from this volume
            volume_us_slices = []
            volume_warped_slices = []
            
            # Store metadata for each slice to support slice selection in GUI
            slice_metadata = {}
            
            # Process each slice with the requested step size
            # Calculate the valid slice indices to avoid out-of-range errors
            max_slice_idx = labelmap.shape[2] - 1
            slice_indices = list(range(0, labelmap.shape[2], step_size))
            # Ensure we don't exceed the maximum valid index
            slice_indices = [idx for idx in slice_indices if idx <= max_slice_idx]
            
            for slice_idx in tqdm.tqdm(slice_indices, desc="US slice rendering"):
                # Extract and prepare the slice
                slice_data = labelmap[:, :, slice_idx].astype(np.int64)
                
                # Convert to tensor and resize/crop
                slice_tensor = torch.as_tensor(slice_data, dtype=torch.uint8, device=self.device)
                
                # Resize and center crop
                slice_tensor_resized = F.interpolate(
                    slice_tensor.unsqueeze(0).unsqueeze(0).float(), 
                    size=(self.resize_size, self.resize_size), 
                    mode='nearest'
                )
                
                # Center crop
                crop_top = (self.resize_size - self.crop_size) // 2
                crop_left = (self.resize_size - self.crop_size) // 2
                slice_tensor_cropped = slice_tensor_resized[:, :, 
                                                           crop_top:crop_top + self.crop_size, 
                                                           crop_left:crop_left + self.crop_size].squeeze(0).squeeze(0)
                
                # Render ultrasound image using our renderer
                with torch.no_grad():
                    us_slice = self.ultrasound_rendering(slice_tensor_cropped)
                
                # Always store as CPU tensor for consistent behavior
                volume_us_slices.append(us_slice.cpu())
                
                # Save the output image
                us_image_pil = transforms.ToPILImage()(us_slice.cpu().squeeze())
                us_image_pil.save(f"{dest}_{slice_idx}.png")
                
                # Process the warped labels for segmentation
                # Apply the same warping to the label map
                with torch.no_grad():
                    warped_labelmap = self.ultrasound_rendering.warp_img(slice_tensor_cropped)
                    warped_labelmap_cpu = warped_labelmap.cpu().numpy() if hasattr(warped_labelmap, "cpu") else warped_labelmap
                
                # Transpose and flip for proper orientation
                a = np.fliplr(np.asarray(warped_labelmap_cpu).transpose(1, 0))
                
                # Process each tissue type to create individual segmentations
                slice_warped_labels = []
                tag_masks = {}  # Store masks for later use
                
                for tag in [2, 3, 4, 6, 8, 11, 12, 13]:
                    # Create binary mask
                    tag_mask_tensor = torch.tensor((a == tag), dtype=torch.uint8, device=self.device)
                    
                    # Apply morphological operations with PyTorch
                    # This is more reliable than the scipy/cupy operations
                    closed_tensor = binary_closing(tag_mask_tensor, iterations=3, device=self.device)
                    dilated_tensor = binary_dilation(closed_tensor, iterations=2, device=self.device)
                    filled_tensor = binary_fill_holes(dilated_tensor, device=self.device)
                    
                    # Convert back to numpy for storage
                    filled = filled_tensor.cpu().numpy().astype(np.float32)
                    
                    # Store processed mask
                    tag_masks[tag] = filled
                    
                    # Store label info
                    slice_warped_labels.append((filled, self.l_dict[tag]))
                
                # Store metadata for this slice to support UI interaction
                slice_metadata[slice_idx] = {
                    'original_slice': slice_data,
                    'us_image': us_slice.cpu(),
                    'warped_labelmap': warped_labelmap_cpu,
                    'tag_masks': tag_masks,
                    'file_path': f"{dest}_{slice_idx}.png"
                }
                
                volume_warped_slices.append(slice_warped_labels)
            
            # Generate point cloud data for this volume
            temp = torch.as_tensor(labelmap, device=self.device)
            
            # Point palette from the notebook
            pointpalette = [torch.tensor([
                [0, 0, 0, 255],
                [0, 0, 0, 255],
                [220, 30, 30, 255],
                [170, 80, 0, 31],
                [0, 170, 0, 255],
                [0, 0, 0, 255],
                [0, 175, 20, 255],
                [0, 0, 0, 255],
                [0, 170, 190, 255],
                [0, 0, 0, 255],
                [0, 0, 0, 255],
                [0, 120, 230, 255],
                [115, 65, 200, 31],
                [255, 0, 150, 255]
            ]),
            [0, 0, 100000, 100000, 100000, 0, 100000, 0, 100000, 0, 0, 100000, 400000, 100000]]
            
            list_pos = []
            list_color = []
            
            # Sample points for point cloud visualization
            for i, count in enumerate(pointpalette[1]):
                if count != 0:
                    mask = (temp_tensor == i)
                    coords = mask.nonzero(as_tuple=False)
                    
                    if coords.shape[0] > count:
                        perm = torch.randperm(coords.shape[0])[:count]
                        selected = coords[perm]
                    else:
                        selected = coords
                        
                    list_pos.append(selected)
                    color_tensor = torch.zeros((count, 4), dtype=torch.uint8)
                    color_tensor[:] = torch.as_tensor(pointpalette[0][i], dtype=torch.uint8)
                    list_color.append(color_tensor)
            
            pcd_base.append([list_pos, list_color, temp_tensor.shape])
        
        print("US SIMULATION COMPLETED")
        return us_imgs, warped_result, pcd_base

    def sample(self):
        raise NotImplementedError("OptimizedLotusRenderer is not yet implemented.")

    def name():
        return "OptimizedLotusRenderer"

class PointCloudSampler(BaseComponent):
    """Class for sampling and processing point clouds from segmentation maps."""
    
    def __init__(self, kwargs=None):
        """Initialize point cloud sampler.
        
        Args:
            kwargs: Dictionary containing configuration parameters including:
                device: The device to use for computation ('cuda' or 'cpu')
                Other configuration parameters
        """
        # Ensure kwargs is a dictionary
        if kwargs is None:
            kwargs = {}
        elif isinstance(kwargs, str):
            # Handle case where a string is passed instead of a dict
            device_str = kwargs
            kwargs = {'device': device_str}
            
        super().__init__(kwargs.get('device', 'cpu'))
        
        # Store labelmap references for dynamic adjustment
        self.stored_labelmaps = []
        
        # Default point palette (will be overridden by config)
        palette_colors = [
            [0, 0, 0, 255],      # Label 0
            [0, 0, 0, 255],      # Label 1
            [220, 30, 30, 255],  # Label 2 - lung
            [170, 80, 0, 31],    # Label 3 - fat
            [0, 170, 0, 255],    # Label 4 - vessel
            [0, 0, 0, 255],      # Label 5
            [0, 175, 20, 255],   # Label 6 - kidney
            [0, 0, 0, 255],      # Label 7
            [0, 170, 190, 255],  # Label 8 - muscle
            [0, 0, 0, 255],      # Label 9 - background
            [0, 0, 0, 255],      # Label 10
            [0, 120, 230, 255],  # Label 11 - liver
            [115, 65, 200, 31],  # Label 12 - soft tissue
            [255, 0, 150, 255]   # Label 13 - bone
        ]
        
        # Default points per label
        self.points_per_label = [
            0,       # Label 0
            0,       # Label 1 
            100000,  # Label 2 - lung
            100000,  # Label 3 - fat
            100000,  # Label 4 - vessel
            0,       # Label 5
            100000,  # Label 6 - kidney
            0,       # Label 7
            100000,  # Label 8 - muscle
            0,       # Label 9 - background
            0,       # Label 10
            100000,  # Label 11 - liver
            400000,  # Label 12 - soft tissue
            100000   # Label 13 - bone
        ]
        
        # Use config values if provided
        self.pointpalette = kwargs.get('pointpalette', [torch.tensor(palette_colors), self.points_per_label])
    
    def update_points_per_label(self, new_counts: List[int]):
        """Update the number of points to sample for each label.
        
        Args:
            new_counts: List of new point counts for each label
        """
        self.points_per_label = new_counts
        self.pointpalette[1] = new_counts

    def store_labelmaps(self, labelmaps: List[np.ndarray]):
        """Store labelmaps for later use in point cloud generation."""        # Clear previous labelmaps to avoid accumulation

        # Check if input is empty
        if not labelmaps:
            print("Warning: Empty labelmaps list provided")
            # Create a default small labelmap to avoid errors
            self.stored_labelmaps.append([np.zeros((100, 100, 100), dtype=np.uint8)])
            return
        
        # Process and store labelmaps
        r = []

        if isinstance(labelmaps, list):
            for idx, labelmap in enumerate(labelmaps):
                # Convert CuPy array to NumPy if needed
                if hasattr(labelmap, 'get'):
                    numpy_labelmap = labelmap.get()
                else:
                    numpy_labelmap = np.asarray(labelmap)
                
                # Ensure data type is uint8
                numpy_labelmap = numpy_labelmap.astype(np.uint8)
                
                # Check if labelmap has valid shape and values
                if numpy_labelmap.size > 0:
                    r.append(numpy_labelmap)
                    print(f"Stored labelmap {idx} with shape {numpy_labelmap.shape}")
                else:
                    print(f"Warning: Labelmap {idx} is empty, creating default")
                    r.append(np.zeros((100, 100, 100), dtype=np.uint8))

        else:
            # Handle case where a single labelmap is passed
            try:
                # Convert CuPy array to NumPy if needed
                if hasattr(labelmaps, 'get'):
                    numpy_labelmap = labelmaps.get()
                else:
                    numpy_labelmap = np.asarray(labelmaps)
                
                # Ensure data type is uint8
                numpy_labelmap = numpy_labelmap.astype(np.uint8)
                
                # Add to storage
                r.append(numpy_labelmap)
                print(f"Stored single labelmap with shape {numpy_labelmap.shape}")
            except Exception as e:
                print(f"Error processing single labelmap: {e}")
                # Add a default placeholder
                r.append(np.zeros((100, 100, 100), dtype=np.uint8))

        self.stored_labelmaps.append(r)
        
    
    def sample(self, idx: int = 0, settings: Dict = None) -> List:
        """Sample points from label maps to create point clouds.
        
        Args:
            settings: Optional custom settings for point cloud generation
            
        Returns:
            List of point cloud data for each label map
        """
        # Use custom settings if provided
        if settings and 'pointpalette' in settings:
            pointpalette = settings['pointpalette']
        else:
            pointpalette = self.pointpalette
            
        pcds = []

        torch.set_default_device(self.device)
        
        # If there are no stored labelmaps, return a default placeholder
        if not self.stored_labelmaps:
            print("Warning: No labelmaps stored for point cloud generation.")
            # Create a default placeholder with a simple 100x100x100 volume
            # This ensures UI functionality even without real data
            default_shape = (100, 100, 100)
            # Create a simple box shape with multiple colors as a demo
            default_pos = []
            default_color = []
            
            # Generate demo points for a cube
            for i, count in enumerate([0, 0, 10000, 10000, 10000, 0, 10000, 0, 10000, 0, 0, 10000, 0, 10000]):
                if count > 0:
                    # Create some points for this label
                    z_pos = 20 + i * 5  # Position each label at a different z level
                    # Create a small cube of points
                    x = torch.randint(40, 60, (count,), device=self.device)
                    y = torch.randint(40, 60, (count,), device=self.device)
                    z = torch.randint(z_pos - 2, z_pos + 2, (count,), device=self.device)
                    points = torch.stack([x, y, z], dim=1)
                    
                    default_pos.append(points)
                    
                    # Use the palette color for this label
                    color_tensor = torch.zeros((count, 4), dtype=torch.uint8, device=self.device)
                    color_tensor[:] = self.pointpalette[0][i]
                    default_color.append(color_tensor)
            
            # If no points were created, add a single dummy point
            if not default_pos:
                default_pos = [torch.zeros((1, 3), device=self.device)]
                default_color = [torch.tensor([[255, 255, 255, 255]], device=self.device)]
                
            pcds.append([default_pos, default_color, default_shape])
            print("Created demo point cloud with placeholder data")
            return pcds

        # Process each real labelmap
        print(f"Processing {len(self.stored_labelmaps[idx])} labelmaps for point cloud generation")
        for labelmap_idx, labelmap in enumerate(tqdm.tqdm(self.stored_labelmaps[idx], desc="Pointcloud rendering")):
            try:
                # Convert to torch tensor
                temp = torch.as_tensor(labelmap, device=self.device)
                print(f"Labelmap {labelmap_idx} shape: {temp.shape}, unique values: {torch.unique(temp).cpu().numpy()}")
                
                list_pos = []
                list_color = []
                
                # Get unique labels and their counts
                unique_labels = torch.unique(temp).cpu().numpy()
                print(f"Unique labels in labelmap {labelmap_idx}: {unique_labels}")
                
                # Process each label with requested point count
                points_sampled = 0
                for i, count in enumerate(pointpalette[1]):
                    if count != 0 and i in unique_labels:
                        try:
                            # Create mask for this label
                            mask = (temp == i)
                            total_voxels = mask.sum().item()
                            print(f"Label {i} has {total_voxels} voxels")
                            
                            if total_voxels > 0:
                                coords = mask.nonzero(as_tuple=False)
                                
                                # Sample points (either all if less than count, or random subset)
                                actual_count = min(count, total_voxels)
                                if coords.shape[0] > actual_count:
                                    perm = torch.randperm(coords.shape[0])[:actual_count]
                                    selected = coords[perm]
                                else:
                                    selected = coords
                                
                                points_sampled += selected.shape[0]
                                list_pos.append(selected)
                                
                                # Assign colors
                                color_tensor = torch.zeros((selected.shape[0], 4), dtype=torch.uint8, device=self.device)
                                color_tensor[:] = torch.as_tensor(pointpalette[0][i], dtype=torch.uint8)
                                list_color.append(color_tensor)
                                print(f"Added {selected.shape[0]} points for label {i}")
                        except Exception as e:
                            print(f"Error processing label {i}: {e}")
                
                print(f"Total points sampled: {points_sampled}")
                
                # Include shape info
                pcd_result = [list_pos, list_color, temp.shape]
                
                # If this labelmap has metadata, include it
                if hasattr(labelmap, 'slice_metadata'):
                    pcd_result.append(labelmap.slice_metadata)
                
                pcds = pcd_result
                print(f"Completed point cloud for labelmap {labelmap_idx}")
                
            except Exception as e:
                print(f"Error processing labelmap {labelmap_idx}: {e}")
                import traceback
                traceback.print_exc()
                
                # Add a default placeholder in case of error
                default_shape = temp.shape if 'temp' in locals() else (100, 100, 100)
                default_pos = [torch.zeros((1, 3), device=self.device)]
                default_color = [torch.tensor([[255, 255, 255, 255]], device=self.device)]
                pcds = [default_pos, default_color, default_shape]
        
        print(f"Generated {len(pcds)} point clouds")
        return pcds
    
        
    def add_axis_pcd(self, base, y, axis_sample_size=20000, axis_points_rgba=[255,255,255,255]) -> tri.PointCloud:
        """Create a point cloud with a highlighted slice at position y.
        
        Args:
            base: Base point cloud data [points, colors, shape]
            y: Slice index to highlight
            axis_sample_size: Number of points to sample for the axis indicator
            axis_points_rgba: Color for the axis indicator points
            
        Returns:
            A PointCloud object with the slice highlighted
        """
        # Extract points, colors, and shape from base
        points, colors, shape = base
        
        # Implement EXACTLY as in the reference code
        pcd_points = points.copy()
        pcd_colors = colors.copy()

        axis = torch.zeros(shape, device=self.device)
        axis[:,:,y] = 1
        tmp = axis.nonzero()

        idx_list = list(range(tmp.shape[0]))
        select_idx = np.random.choice(idx_list, size=axis_sample_size)
        pcd_points.append(tmp[select_idx])

        tmp = torch.zeros((axis_sample_size, 4), dtype=torch.uint8, device=self.device)
        tmp[:,...] = torch.as_tensor(axis_points_rgba)
        pcd_colors.append(tmp)

        point_pos = torch.concat(pcd_points)
        colors = torch.concat(pcd_colors)

        point_pos = point_pos.float()

        point_displacement = torch.rand(point_pos.shape).to(self.device)
        point_pos += point_displacement

        shape = torch.FloatTensor(np.array(shape)).to(self.device)
        point_pos /= shape
        point_pos = point_pos - .5

        pcd = tri.PointCloud(point_pos.cpu(), colors.cpu()).apply_transform(
                    np.dot(
                        tri.transformations.rotation_matrix(np.pi, [1, 0, 0]),
                        tri.transformations.rotation_matrix(np.pi/2, [0, -1, 0])
                    )
            )

        return pcd

        
    def get_label_counts(self):
        """
        Get the number of voxels for each label in the stored labelmaps.
        
        Returns:
            Dictionary mapping label indices to voxel counts
        """
        if not self.stored_labelmaps:
            return {}
        
        result = {}
        for i, labelmap in enumerate(self.stored_labelmaps[0]):
            if not isinstance(labelmap, torch.Tensor):
                tensor = torch.as_tensor(labelmap, device=self.device)
            else:
                tensor = labelmap
                
            # Get unique labels and their counts
            unique_labels, counts = torch.unique(tensor, return_counts=True)
            
            # Convert to CPU for dictionary creation
            if tensor.is_cuda:
                unique_labels = unique_labels.cpu().numpy()
                counts = counts.cpu().numpy()
            else:
                unique_labels = unique_labels.numpy()
                counts = counts.numpy()
                
            # Create dictionary for this labelmap
            label_counts = {int(label): int(count) for label, count in zip(unique_labels, counts)}
            result[i] = label_counts
            
        return result
    
    def name():
        return "default"