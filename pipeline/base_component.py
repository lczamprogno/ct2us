try:
    import cupy as cp
    import cupyx.scipy.ndimage as cusci
except ImportError:
    print("Error loading cupy and cusci, GPU not available?")

from abc import ABC, abstractmethod
import scipy.ndimage
import scipy
import numpy as np
import torch

from itertools import islice

def dict_2_map(d: dict[str, int]) -> list[list[int]]:
    """
    Convert dictionary of string keys to integer values into a mapping list.
    
    Args:
        d: Dictionary with string keys and integer values
        
    Returns:
        A list of lists where each inner list contains the keys for a specific value
    """
    map_list = [[] for _ in range(15)]

    for k, v in d.items():
        int_k = int(k)
        map_list[v].append(int_k)

    return map_list

def batched(iterable, n):
    """
    Batch data into tuples of length n. The last batch may be shorter.
    
    Args:
        iterable: The iterable to be batched
        n: Batch size
        
    Returns:
        Yields batches of the data
    """
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch

class BaseComponent(ABC):
    """Base class for all pipeline components"""
    
    def __init__(self, device_str):
        """
        Initialize the base component.
        
        Args:
            device: The device to use (CPU or CUDA)
        """        
        # Set up the appropriate array library and operations based on device
        self.config = {}

        # Validate and normalize device string
        if isinstance(device_str, str):
            device_str = device_str.lower()
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        
        # Set appropriate array module based on device and CUDA availability
        if device_str != 'cpu' and cuda_available:
            try:
                import cupy as cp
                import cupyx.scipy.ndimage as cusci
                self.m = cp
                self.ops = cusci
            except ImportError:
                self.m = np
                self.ops = scipy.ndimage
        else:
            self.m = np
            self.ops = scipy.ndimage
            if device_str == 'cuda' and not cuda_available:
                print("CUDA requested but not available. Using CPU instead.")

        # Set PyTorch device
        if device_str != 'cpu' and cuda_available:
            self.device = torch.device("cuda", 0)
        else:
            self.device = torch.device("cpu")

    @abstractmethod
    def name() -> str:
        """Return the name of the component"""
        pass
    
    """"
    def dilation(self, imgs: torch.Tensor, kernel_size: int=3, iterations: int=1) -> torch.Tensor:
        '''Perform binary dilation on a tensor.'''
        if not isinstance(iterations, int):
            iterations = int(iterations)
        
        # Convert to tensor if needed
        if not isinstance(imgs, torch.Tensor):
            if self.m == cp and hasattr(imgs, 'get'):
                # Convert from CuPy array
                imgs_tensor = torch.as_tensor(imgs.get(), device=self.device)
            else:
                # Convert from NumPy array
                imgs_tensor = torch.as_tensor(imgs, device=self.device)
        else:
            imgs_tensor = imgs.to(self.device)
            
        # Use PyTorch-based morphological operations
        from pipeline.torch_morphology import binary_dilation
        
        with torch.no_grad():
            d_imgs = binary_dilation(imgs_tensor, kernel_size, iterations, self.device)
        
        # Convert back to original format if needed
        if not isinstance(imgs, torch.Tensor):
            if self.m == cp:
                # Convert back to CuPy
                return cp.asarray(d_imgs.cpu().numpy())
            else:
                # Convert back to NumPy
                return d_imgs.cpu().numpy()
        
        return d_imgs

    def erosion(self, imgs: torch.Tensor, kernel_size: int=3, iterations: int=1) -> torch.Tensor:
        '''
        Perform binary erosion on a tensor.
        
        Args:
            imgs: Input tensor
            kernel_size: Siz    def erosion(self, imgs: torch.Tensor, kernel_size: int=3, iterations: int=1) -> torch.Tensor:
        '''Perform binary erosion on a tensor.'''
        if not isinstance(iterations, int):
            iterations = int(iterations)
            
        # Convert to tensor if needed
        if not isinstance(imgs, torch.Tensor):
            if self.m == cp and hasattr(imgs, 'get'):
                # Convert from CuPy array
                imgs_tensor = torch.as_tensor(imgs.get(), device=self.device)
            else:
                # Convert from NumPy array
                imgs_tensor = torch.as_tensor(imgs, device=self.device)
        else:
            imgs_tensor = imgs.to(self.device)
        
        # Use PyTorch-based morphological operations
        from pipeline.torch_morphology import binary_erosion
        
        with torch.no_grad():
            d_imgs = binary_erosion(imgs_tensor, kernel_size, iterations, self.device)
        
        # Convert back to original format if needed
        if not isinstance(imgs, torch.Tensor):
            if self.m == cp:
                # Convert back to CuPy
                return cp.asarray(d_imgs.cpu().numpy())
            else:
                # Convert back to NumPy
                return d_imgs.cpu().numpy()
        
        return d_imgs
        """