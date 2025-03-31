"""
PyTorch-based morphological operations with Numba JIT optimization.

This module provides GPU-accelerated morphological operations using PyTorch tensors
and Numba JIT compilation for CPU fallback.
"""

import torch
import torch.nn.functional as F
import numba
import numpy as np
from typing import Tuple, Union, Optional

# # Helper functions for PyTorch morphological operations
# def binary_closing_torch(tensor, iterations=1, device=None):
#     """Apply binary closing using PyTorch operations"""
#     # Get device from tensor if not provided
#     if device is None:
#         device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#     # Convert to tensor if not already
#     if not isinstance(tensor, torch.Tensor):
#         if isinstance(tensor, np.ndarray):
#             tensor = torch.from_numpy(tensor).to(device)
#         elif hasattr(tensor, 'get'):  # CuPy array
#             tensor = torch.from_numpy(tensor.get()).to(device)
            
#     tensor = tensor.float()  # Ensure float for convolution
    
#     # Create circular kernel
#     kernel_size = 3
#     radius = kernel_size // 2
#     y, x = torch.meshgrid(
#         torch.arange(-radius, radius + 1, device=device),
#         torch.arange(-radius, radius + 1, device=device),
#         indexing='ij'
#     )
#     kernel = ((x ** 2 + y ** 2) <= radius ** 2).float()
#     kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    
#     # Handle different tensor dimensions
#     original_shape = tensor.shape
#     if len(original_shape) == 2:
#         tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    
#     # Apply dilation then erosion
#     # Dilation
#     for _ in range(iterations):
#         tensor = (F.conv2d(tensor, kernel, padding=radius) > 0).float()
    
#     # Erosion
#     kernel_sum = kernel.sum()
#     for _ in range(iterations):
#         tensor = (F.conv2d(tensor, kernel, padding=radius) >= kernel_sum).float()
        
#     # Restore original shape
#     if len(original_shape) == 2:
#         tensor = tensor.squeeze(0).squeeze(0)
        
#     return tensor

# def binary_dilation_torch(tensor, iterations=1, device=None):
#     """Apply binary dilation using PyTorch operations"""
#     # Get device from tensor if not provided
#     if device is None:
#         device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#     # Convert to tensor if not already
#     if not isinstance(tensor, torch.Tensor):
#         if isinstance(tensor, np.ndarray):
#             tensor = torch.from_numpy(tensor).to(device)
#         elif hasattr(tensor, 'get'):  # CuPy array
#             tensor = torch.from_numpy(tensor.get()).to(device)
            
#     tensor = tensor.float()  # Ensure float for convolution
    
#     # Create circular kernel
#     kernel_size = 3
#     radius = kernel_size // 2
#     y, x = torch.meshgrid(
#         torch.arange(-radius, radius + 1, device=device),
#         torch.arange(-radius, radius + 1, device=device),
#         indexing='ij'
#     )
#     kernel = ((x ** 2 + y ** 2) <= radius ** 2).float()
#     kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    
#     # Handle different tensor dimensions
#     original_shape = tensor.shape
#     if len(original_shape) == 2:
#         tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    
#     # Apply dilation
#     for _ in range(iterations):
#         tensor = (F.conv2d(tensor, kernel, padding=radius) > 0).float()
        
#     # Restore original shape
#     if len(original_shape) == 2:
#         tensor = tensor.squeeze(0).squeeze(0)
        
#     return tensor

# def binary_fill_holes_torch(tensor, device=None):
#     """Fill holes in binary mask using PyTorch operations"""
#     # Get device from tensor if not provided
#     if device is None:
#         device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#     # Convert to tensor if not already
#     if not isinstance(tensor, torch.Tensor):
#         if isinstance(tensor, np.ndarray):
#             tensor = torch.from_numpy(tensor).to(device)
#         elif hasattr(tensor, 'get'):  # CuPy array
#             tensor = torch.from_numpy(tensor.get()).to(device)
    
#     # Need to move to CPU for hole filling algorithm
#     mask_np = tensor.cpu().numpy().astype(np.uint8)
    
#     # Create padded mask for flood fill
#     h, w = mask_np.shape
#     padded = np.zeros((h+2, w+2), dtype=np.uint8)
#     padded[1:-1, 1:-1] = mask_np
    
#     # Fill from outside boundary
#     temp = padded.copy()
#     stack = [(0, 0)]
#     temp[0, 0] = 2  # Mark as filled
    
#     while stack:
#         y, x = stack.pop()
#         for ny, nx in [(y+1, x), (y-1, x), (y, x+1), (y, x-1)]:
#             if 0 <= ny < h+2 and 0 <= nx < w+2 and temp[ny, nx] == 0:
#                 temp[ny, nx] = 2  # Mark as visited
#                 stack.append((ny, nx))
    
#     # Holes are pixels not reached by outside fill
#     result = mask_np.copy()
#     for i in range(h):
#         for j in range(w):
#             if temp[i+1, j+1] == 0:  # If not reached, it's a hole
#                 result[i, j] = 1
    
#     return torch.from_numpy(result).to(device).float()

# Numba optimized CPU implementations for fallback
@numba.jit(nopython=True, parallel=True)
def _binary_dilation_3d_numba(input_array: np.ndarray, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Numba-accelerated binary dilation implementation for 3D arrays.
    
    Args:
        input_array: 3D numpy array to dilate
        kernel: 3D structuring element
        iterations: Number of times to apply the dilation
        
    Returns:
        Dilated 3D array
    """
    result = input_array.copy()
    temp = np.zeros_like(result)
    
    # Get kernel dimensions
    k_depth, k_height, k_width = kernel.shape
    k_depth_half, k_height_half, k_width_half = k_depth // 2, k_height // 2, k_width // 2
    
    # Get input dimensions
    depth, height, width = input_array.shape
    
    for _ in range(iterations):
        # Reset temp array
        temp.fill(0)
        
        # Perform dilation
        for z in numba.prange(depth):
            for y in range(height):
                for x in range(width):
                    if result[z, y, x] > 0:
                        # Apply kernel at each foreground voxel
                        for kz in range(k_depth):
                            z_pos = z + kz - k_depth_half
                            if z_pos < 0 or z_pos >= depth:
                                continue
                                
                            for ky in range(k_height):
                                y_pos = y + ky - k_height_half
                                if y_pos < 0 or y_pos >= height:
                                    continue
                                    
                                for kx in range(k_width):
                                    x_pos = x + kx - k_width_half
                                    if x_pos < 0 or x_pos >= width:
                                        continue
                                        
                                    if kernel[kz, ky, kx] > 0:
                                        temp[z_pos, y_pos, x_pos] = 1
        
        # Update result for next iteration
        result = temp.copy()
    
    return result

@numba.jit(nopython=True, parallel=True)
def _binary_erosion_3d_numba(input_array: np.ndarray, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Numba-accelerated binary erosion implementation for 3D arrays.
    
    Args:
        input_array: 3D numpy array to erode
        kernel: 3D structuring element
        iterations: Number of times to apply the erosion
        
    Returns:
        Eroded 3D array
    """
    result = input_array.copy()
    temp = np.zeros_like(result)
    
    # Get kernel dimensions
    k_depth, k_height, k_width = kernel.shape
    k_depth_half, k_height_half, k_width_half = k_depth // 2, k_height // 2, k_width // 2
    
    # Get input dimensions
    depth, height, width = input_array.shape
    
    for _ in range(iterations):
        # Reset temp array
        temp.fill(0)
        
        # Perform erosion
        for z in numba.prange(depth):
            for y in range(height):
                for x in range(width):
                    # Check if all kernel elements are satisfied
                    valid = True
                    
                    for kz in range(k_depth):
                        z_pos = z + kz - k_depth_half
                        if z_pos < 0 or z_pos >= depth:
                            if kernel[kz, ky, kx] > 0:
                                valid = False
                                break
                            continue
                            
                        for ky in range(k_height):
                            y_pos = y + ky - k_height_half
                            if y_pos < 0 or y_pos >= height:
                                if kernel[kz, ky, kx] > 0:
                                    valid = False
                                    break
                                continue
                                
                            for kx in range(k_width):
                                x_pos = x + kx - k_width_half
                                if x_pos < 0 or x_pos >= width:
                                    if kernel[kz, ky, kx] > 0:
                                        valid = False
                                        break
                                    continue
                                    
                                if kernel[kz, ky, kx] > 0 and result[z_pos, y_pos, x_pos] == 0:
                                    valid = False
                                    break
                            
                            if not valid:
                                break
                        
                        if not valid:
                            break
                    
                    if valid:
                        temp[z, y, x] = 1
        
        # Update result for next iteration
        result = temp.copy()
    
    return result

@numba.jit(nopython=True)
def _connected_components_3d_numba(input_array: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Numba-accelerated 3D connected components labeling for binary images.
    
    Args:
        input_array: 3D binary numpy array
        
    Returns:
        Tuple containing (labeled array, number of components)
    """
    depth, height, width = input_array.shape
    output = np.zeros_like(input_array, dtype=np.int32)
    
    # Initialize label counter
    current_label = 1
    
    # Create an equivalence table for labels
    max_labels = depth * height * width // 4  # Estimate of max possible labels
    equivalences = np.zeros((max_labels, 2), dtype=np.int32)
    equiv_count = 0
    
    # First pass: assign initial labels and record equivalences
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                if input_array[z, y, x] == 0:
                    continue
                
                # Check 6-connected neighbors (only those already processed)
                neighbors = []
                
                if z > 0 and output[z-1, y, x] > 0:
                    neighbors.append(output[z-1, y, x])
                
                if y > 0 and output[z, y-1, x] > 0:
                    neighbors.append(output[z, y-1, x])
                
                if x > 0 and output[z, y, x-1] > 0:
                    neighbors.append(output[z, y, x-1])
                
                if not neighbors:
                    # No neighbors, assign new label
                    output[z, y, x] = current_label
                    current_label += 1
                else:
                    # Use minimum neighbor label
                    min_label = min(neighbors)
                    output[z, y, x] = min_label
                    
                    # Record equivalences if needed
                    for n in neighbors:
                        if n != min_label:
                            # Add to equivalence table
                            equivalences[equiv_count, 0] = min_label
                            equivalences[equiv_count, 1] = n
                            equiv_count += 1
    
    # Resolve equivalences (simple union-find)
    if equiv_count > 0:
        # Extract actual equivalences
        actual_equivalences = equivalences[:equiv_count]
        
        # Create label map for second pass
        label_map = np.arange(current_label, dtype=np.int32)
        
        # Process equivalences until no more changes
        while True:
            changes = 0
            for i in range(equiv_count):
                a, b = actual_equivalences[i]
                a_root = label_map[a]
                b_root = label_map[b]
                
                if a_root != b_root:
                    # Merge labels (always to smaller value)
                    min_root = min(a_root, b_root)
                    max_root = max(a_root, b_root)
                    
                    # Update all labels that point to max_root
                    for j in range(1, current_label):
                        if label_map[j] == max_root:
                            label_map[j] = min_root
                            changes += 1
            
            if changes == 0:
                break
        
        # Second pass: apply resolved labels
        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    if output[z, y, x] > 0:
                        output[z, y, x] = label_map[output[z, y, x]]
    
    # Count final number of components
    final_labels = np.unique(output)
    num_components = len(final_labels) - (1 if 0 in final_labels else 0)
    
    return output, num_components

# PyTorch implementations
def binary_dilation(input_tensor: torch.Tensor, kernel_size: int = 3, 
                   iterations: int = 1, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    PyTorch-based binary dilation for 2D or 3D tensors.
    
    Args:
        input_tensor: Input tensor (batch_size, channels, [depth,] height, width)
        kernel_size: Size of the dilation kernel (single number for cubic/square kernel)
        iterations: Number of dilation iterations
        device: Device to run the operation on (defaults to input_tensor's device)
        
    Returns:
        Dilated tensor of the same shape as input
    """
    if device is None:
        device = input_tensor.device
    
    # Move tensor to target device if needed
    input_tensor = input_tensor.to(device)
    
    # Check if we're processing a 3D tensor
    is_3d = len(input_tensor.shape) == 5 or (len(input_tensor.shape) == 4 and input_tensor.shape[1] == 1) or len(input_tensor.shape) == 3
    
    # Use max pooling which is equivalent to dilation
    if is_3d:
        # Handle 3D cases with various dimensions
        if len(input_tensor.shape) == 3:
            # If we have a 3D volume without batch dimension, add batch and channel dims
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
        elif len(input_tensor.shape) == 4:
            # If we have batch + 3D volume, add channel dim
            input_tensor = input_tensor.unsqueeze(1)
        
        # Create kernel with appropriate padding
        padding = kernel_size // 2
        
        # Apply iterations
        result = input_tensor
        for _ in range(iterations):
            result = F.max_pool3d(result, 
                                 kernel_size=kernel_size, 
                                 stride=1, 
                                 padding=padding)
        
        # If we added a dimension earlier, remove it now
        if len(input_tensor.shape) == 4:
            result = result.squeeze(1)
    else:
        # Handle 2D input
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(1)
        
        # Create kernel with appropriate padding
        padding = kernel_size // 2
        
        # Apply iterations
        result = input_tensor
        for _ in range(iterations):
            result = F.max_pool2d(result, 
                                 kernel_size=kernel_size, 
                                 stride=1, 
                                 padding=padding)
        
        # If we added a dimension earlier, remove it now
        if len(input_tensor.shape) == 3:
            result = result.squeeze(1)
    
    # Ensure binary output (0 or 1)
    return (result > 0).to(torch.uint8)

def binary_erosion(input_tensor: torch.Tensor, kernel_size: int = 3, 
                  iterations: int = 1, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    PyTorch-based binary erosion for 2D or 3D tensors.
    
    Args:
        input_tensor: Input tensor (batch_size, channels, [depth,] height, width)
        kernel_size: Size of the erosion kernel (single number for cubic/square kernel)
        iterations: Number of erosion iterations
        device: Device to run the operation on (defaults to input_tensor's device)
        
    Returns:
        Eroded tensor of the same shape as input
    """
    if device is None:
        device = input_tensor.device
    
    # Move tensor to target device if needed
    input_tensor = input_tensor.to(device)
    
    # Check if we're processing a 3D tensor
    is_3d = len(input_tensor.shape) == 5 or (len(input_tensor.shape) == 4 and input_tensor.shape[1] == 1) or len(input_tensor.shape) == 3
    
    # For erosion, we invert, dilate, then invert again
    inverted = (input_tensor == 0).to(torch.uint8)
    
    # Use dilation on the inverted image
    if is_3d:
        # Handle 3D cases with various dimensions
        if len(inverted.shape) == 3:
            # If we have a 3D volume without batch dimension, add batch and channel dims
            inverted = inverted.unsqueeze(0).unsqueeze(0)
        elif len(inverted.shape) == 4:
            # If we have batch + 3D volume, add channel dim
            inverted = inverted.unsqueeze(1)
        
        # Create kernel with appropriate padding
        padding = kernel_size // 2
        
        # Apply iterations
        result = inverted
        for _ in range(iterations):
            result = F.max_pool3d(result, 
                                 kernel_size=kernel_size, 
                                 stride=1, 
                                 padding=padding)
        
        # If we added a dimension earlier, remove it now
        if len(inverted.shape) == 4:
            result = result.squeeze(1)
    else:
        # Handle 2D input
        if len(inverted.shape) == 3:
            inverted = inverted.unsqueeze(1)
        
        # Create kernel with appropriate padding
        padding = kernel_size // 2
        
        # Apply iterations
        result = inverted
        for _ in range(iterations):
            result = F.max_pool2d(result, 
                                 kernel_size=kernel_size, 
                                 stride=1, 
                                 padding=padding)
        
        # If we added a dimension earlier, remove it now
        if len(inverted.shape) == 3:
            result = result.squeeze(1)
    
    # Invert again to get erosion
    result = (result == 0).to(torch.uint8)
    
    return result

def binary_closing(input_tensor: torch.Tensor, kernel_size: int = 3, 
                  iterations: int = 1, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    PyTorch-based binary closing (dilation followed by erosion) for 2D or 3D tensors.
    
    Args:
        input_tensor: Input tensor (batch_size, channels, [depth,] height, width)
        kernel_size: Size of the closing kernel (single number for cubic/square kernel)
        iterations: Number of closing iterations
        device: Device to run the operation on (defaults to input_tensor's device)
        
    Returns:
        Closed tensor of the same shape as input
    """
    if device is None:
        device = input_tensor.device
    
    # Move tensor to target device if needed
    input_tensor = input_tensor.to(device)
    
    # Binary closing is dilation followed by erosion
    dilated = binary_dilation(input_tensor, kernel_size, iterations, device)
    closed = binary_erosion(dilated, kernel_size, iterations, device)
    
    return closed

def binary_opening(input_tensor: torch.Tensor, kernel_size: int = 3, 
                  iterations: int = 1, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    PyTorch-based binary opening (erosion followed by dilation) for 2D or 3D tensors.
    
    Args:
        input_tensor: Input tensor (batch_size, channels, [depth,] height, width)
        kernel_size: Size of the opening kernel (single number for cubic/square kernel)
        iterations: Number of opening iterations
        device: Device to run the operation on (defaults to input_tensor's device)
        
    Returns:
        Opened tensor of the same shape as input
    """
    if device is None:
        device = input_tensor.device
    
    # Move tensor to target device if needed
    input_tensor = input_tensor.to(device)
    
    # Binary opening is erosion followed by dilation
    eroded = binary_erosion(input_tensor, kernel_size, iterations, device)
    opened = binary_dilation(eroded, kernel_size, iterations, device)
    
    return opened

def binary_fill_holes(input_tensor: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    PyTorch-based hole filling for 2D or 3D binary tensors.
    
    Args:
        input_tensor: Input tensor (batch_size, [channels,] [depth,] height, width)
        device: Device to run the operation on (defaults to input_tensor's device)
        
    Returns:
        Tensor with holes filled
    """
    if device is None:
        device = input_tensor.device
    
    # Move tensor to target device if needed
    input_tensor = input_tensor.to(device)
    
    # If we need to process it on CPU due to limitations
    if device.type == 'cpu' or not torch.cuda.is_available():
        # Convert to numpy for processing
        if input_tensor.is_cuda:
            cpu_tensor = input_tensor.cpu()
        else:
            cpu_tensor = input_tensor
            
        # Process each batch and channel separately
        if len(cpu_tensor.shape) == 5:  # 3D with batch and channel
            b, c, d, h, w = cpu_tensor.shape
            result = torch.zeros_like(cpu_tensor)
            
            for batch_idx in range(b):
                for channel_idx in range(c):
                    # Get slice and convert to numpy
                    slice_np = cpu_tensor[batch_idx, channel_idx].numpy()
                    
                    # Create a larger array with zeros at the border
                    mask_dilated = np.zeros((d+2, h+2, w+2), dtype=np.uint8)
                    mask_dilated[1:-1, 1:-1, 1:-1] = slice_np
                    
                    # Find background markers (connected regions start at borders)
                    background = np.zeros_like(mask_dilated)
                    background[0, :, :] = 1 - mask_dilated[0, :, :]
                    background[-1, :, :] = 1 - mask_dilated[-1, :, :]
                    background[:, 0, :] = 1 - mask_dilated[:, 0, :]
                    background[:, -1, :] = 1 - mask_dilated[:, -1, :]
                    background[:, :, 0] = 1 - mask_dilated[:, :, 0]
                    background[:, :, -1] = 1 - mask_dilated[:, :, -1]
                    
                    # Use numba optimized connected components to identify background
                    labeled_bg, _ = _connected_components_3d_numba(background)
                    
                    # Background is all labeled components
                    bg_mask = (labeled_bg > 0)
                    
                    # Initial background mask (holes included)
                    background_with_holes = np.zeros_like(mask_dilated)
                    
                    # Fill background via simple connected components propagation
                    for i in range(10):  # Limited iterations for propagation
                        # Dilate current background
                        kernel = np.ones((3, 3, 3), dtype=np.uint8)
                        bg_dilated = _binary_dilation_3d_numba(bg_mask, kernel, iterations=1)
                        
                        # Only accept background in non-object areas
                        new_bg = bg_dilated & (mask_dilated == 0)
                        
                        # If no changes, we're done
                        if np.array_equal(new_bg, bg_mask):
                            break
                            
                        bg_mask = new_bg
                    
                    # Final foreground with holes filled is inverse of background in original region
                    filled = 1 - bg_mask[1:-1, 1:-1, 1:-1]
                    
                    # Set result
                    result[batch_idx, channel_idx] = torch.from_numpy(filled)
            
            # Move back to original device
            return result.to(device)
            
        elif len(cpu_tensor.shape) == 4:  # 3D with batch
            b, d, h, w = cpu_tensor.shape
            result = torch.zeros_like(cpu_tensor)
            
            for batch_idx in range(b):
                # Get slice and convert to numpy
                slice_np = cpu_tensor[batch_idx].numpy()
                
                # Create a larger array with zeros at the border
                mask_dilated = np.zeros((d+2, h+2, w+2), dtype=np.uint8)
                mask_dilated[1:-1, 1:-1, 1:-1] = slice_np
                
                # Find background markers (connected regions start at borders)
                background = np.zeros_like(mask_dilated)
                background[0, :, :] = 1 - mask_dilated[0, :, :]
                background[-1, :, :] = 1 - mask_dilated[-1, :, :]
                background[:, 0, :] = 1 - mask_dilated[:, 0, :]
                background[:, -1, :] = 1 - mask_dilated[:, -1, :]
                background[:, :, 0] = 1 - mask_dilated[:, :, 0]
                background[:, :, -1] = 1 - mask_dilated[:, :, -1]
                
                # Use numba optimized connected components to identify background
                labeled_bg, _ = _connected_components_3d_numba(background)
                
                # Background is all labeled components
                bg_mask = (labeled_bg > 0)
                
                # Fill background via simple connected components propagation
                for i in range(10):  # Limited iterations for propagation
                    # Dilate current background
                    kernel = np.ones((3, 3, 3), dtype=np.uint8)
                    bg_dilated = _binary_dilation_3d_numba(bg_mask, kernel, iterations=1)
                    
                    # Only accept background in non-object areas
                    new_bg = bg_dilated & (mask_dilated == 0)
                    
                    # If no changes, we're done
                    if np.array_equal(new_bg, bg_mask):
                        break
                        
                    bg_mask = new_bg
                
                # Final foreground with holes filled is inverse of background in original region
                filled = 1 - bg_mask[1:-1, 1:-1, 1:-1]
                
                # Set result
                result[batch_idx] = torch.from_numpy(filled)
            
            # Move back to original device
            return result.to(device)
            
        else:  # 2D case
            # Not implemented for simplicity, but would follow similar approach
            raise NotImplementedError("2D fill holes not implemented")
            
    else:
        # GPU implementation using PyTorch
        # For simplicity, we'll use a limited hole filling approach on GPU
        # This uses flood filling from the borders
        
        # Make sure input is binary
        binary_input = (input_tensor > 0).to(torch.uint8)
        
        # Create border mask (for 3D and 2D cases)
        if len(binary_input.shape) == 5:  # 5D case: batch, channel, depth, height, width
            b, c, d, h, w = binary_input.shape
            border_mask = torch.zeros_like(binary_input)
            
            # Set borders to 1
            border_mask[:, :, 0, :, :] = 1
            border_mask[:, :, -1, :, :] = 1
            border_mask[:, :, :, 0, :] = 1
            border_mask[:, :, :, -1, :] = 1
            border_mask[:, :, :, :, 0] = 1
            border_mask[:, :, :, :, -1] = 1
            
        elif len(binary_input.shape) == 4:  # 4D case: batch, depth, height, width
            b, d, h, w = binary_input.shape
            border_mask = torch.zeros_like(binary_input)
            
            # Set borders to 1
            border_mask[:, 0, :, :] = 1
            border_mask[:, -1, :, :] = 1
            border_mask[:, :, 0, :] = 1
            border_mask[:, :, -1, :] = 1
            border_mask[:, :, :, 0] = 1
            border_mask[:, :, :, -1] = 1
            
        else:  # Other cases (batch, height, width) or (height, width)
            if len(binary_input.shape) == 3:  # 3D case: batch, height, width
                b, h, w = binary_input.shape
                border_mask = torch.zeros_like(binary_input)
                
                # Set borders to 1
                border_mask[:, 0, :] = 1
                border_mask[:, -1, :] = 1
                border_mask[:, :, 0] = 1
                border_mask[:, :, -1] = 1
                
            else:  # 2D case: height, width
                h, w = binary_input.shape
                border_mask = torch.zeros_like(binary_input)
                
                # Set borders to 1
                border_mask[0, :] = 1
                border_mask[-1, :] = 1
                border_mask[:, 0] = 1
                border_mask[:, -1] = 1
        
        # Start with border pixels that are background
        current_mask = border_mask & (binary_input == 0)
        
        # Iteratively dilate background from borders
        kernel_size = 3
        for _ in range(max(d, h, w) // 2 if 'd' in locals() else max(h, w) // 2):
            # Dilate current background
            dilated = binary_dilation(current_mask, kernel_size, 1, device)
            
            # Only keep background pixels (not object)
            new_mask = dilated & (binary_input == 0)
            
            # Check if we've converged
            if torch.all(new_mask == current_mask):
                break
                
            current_mask = new_mask
        
        # Final result is original object plus filled holes
        filled = binary_input | ((current_mask == 0) & (binary_input == 0))
        
        return filled

def connected_components(input_tensor: torch.Tensor, 
                         connectivity: int = 6, 
                         device: Optional[torch.device] = None) -> Tuple[torch.Tensor, int]:
    """
    PyTorch-based connected components labeling for binary tensors.
    
    Args:
        input_tensor: Input binary tensor 
        connectivity: Connectivity type (6 or 26 for 3D, 4 or 8 for 2D)
        device: Device to run the operation on (defaults to input_tensor's device)
        
    Returns:
        Tuple containing (labeled tensor, number of components)
    """
    if device is None:
        device = input_tensor.device
    
    # If CUDA isn't available or we're on CPU, use Numba implementation
    if device.type == 'cpu' or not torch.cuda.is_available():
        # Convert to numpy for processing
        if input_tensor.is_cuda:
            cpu_tensor = input_tensor.cpu()
        else:
            cpu_tensor = input_tensor
            
        # Get input shape for processing strategy
        if len(cpu_tensor.shape) == 5:  # 5D: batch, channel, depth, height, width
            b, c, d, h, w = cpu_tensor.shape
            result = torch.zeros((b, c, d, h, w), dtype=torch.int32, device=cpu_tensor.device)
            
            # Process each batch and channel
            for batch_idx in range(b):
                for channel_idx in range(c):
                    # Get binary mask as numpy array
                    binary_mask = cpu_tensor[batch_idx, channel_idx].numpy().astype(np.uint8)
                    
                    # Run connected components
                    labeled, num_components = _connected_components_3d_numba(binary_mask)
                    
                    # Store result
                    result[batch_idx, channel_idx] = torch.from_numpy(labeled)
                    
            # Move result to target device
            return result.to(device), num_components
            
        elif len(cpu_tensor.shape) == 4:  # 4D: batch, depth, height, width
            b, d, h, w = cpu_tensor.shape
            result = torch.zeros((b, d, h, w), dtype=torch.int32, device=cpu_tensor.device)
            
            # Process each batch
            for batch_idx in range(b):
                # Get binary mask as numpy array
                binary_mask = cpu_tensor[batch_idx].numpy().astype(np.uint8)
                
                # Run connected components
                labeled, num_components = _connected_components_3d_numba(binary_mask)
                
                # Store result
                result[batch_idx] = torch.from_numpy(labeled)
                
            # Move result to target device
            return result.to(device), num_components
            
        else:  # 3D: depth, height, width or 2D: height, width
            # Get binary mask as numpy array
            binary_mask = cpu_tensor.numpy().astype(np.uint8)
            
            # Run connected components
            labeled, num_components = _connected_components_3d_numba(binary_mask)
            
            # Move result to target device
            return torch.from_numpy(labeled).to(device), num_components
    
    else:
        # For GPU processing, we need to implement a more efficient algorithm
        # This is a simplified GPU implementation
        
        # Make sure input is binary
        binary_input = (input_tensor > 0).to(torch.uint8)
        
        # Initialize labels with zeros
        labels = torch.zeros_like(binary_input, dtype=torch.int32, device=device)
        
        # Label foreground pixels with unique IDs
        foreground = binary_input > 0
        
        # Create an initial labeling where each pixel has a unique label
        # We'll use a flat index for initial labeling
        if len(binary_input.shape) == 5:  # 5D: batch, channel, depth, height, width
            b, c, d, h, w = binary_input.shape
            for batch_idx in range(b):
                for channel_idx in range(c):
                    mask = foreground[batch_idx, channel_idx]
                    idx = torch.nonzero(mask, as_tuple=True)
                    if len(idx[0]) > 0:
                        # Calculate flat indices
                        flat_idx = idx[0] * h * w + idx[1] * w + idx[2]
                        # Start from 1 to keep 0 as background
                        labels[batch_idx, channel_idx][idx] = flat_idx + 1
        
        elif len(binary_input.shape) == 4:  # 4D: batch, depth, height, width
            b, d, h, w = binary_input.shape
            for batch_idx in range(b):
                mask = foreground[batch_idx]
                idx = torch.nonzero(mask, as_tuple=True)
                if len(idx[0]) > 0:
                    # Calculate flat indices
                    flat_idx = idx[0] * h * w + idx[1] * w + idx[2]
                    # Start from 1 to keep 0 as background
                    labels[batch_idx][idx] = flat_idx + 1
        
        elif len(binary_input.shape) == 3:  # 3D: height, width, depth
            d, h, w = binary_input.shape
            mask = foreground
            idx = torch.nonzero(mask, as_tuple=True)
            if len(idx[0]) > 0:
                # Calculate flat indices
                flat_idx = idx[0] * h * w + idx[1] * w + idx[2]
                # Start from 1 to keep 0 as background
                labels[idx] = flat_idx + 1
        
        else:  # 2D: height, width
            h, w = binary_input.shape
            mask = foreground
            idx = torch.nonzero(mask, as_tuple=True)
            if len(idx[0]) > 0:
                # Calculate flat indices
                flat_idx = idx[0] * w + idx[1]
                # Start from 1 to keep 0 as background
                labels[idx] = flat_idx + 1
        
        # Propagate labels using dilation operations
        # We use this for approximating connected components
        kernel_size = 3
        max_iterations = 50  # Limit iterations for convergence
        
        for _ in range(max_iterations):
            # Store current labels for convergence check
            prev_labels = labels.clone()
            
            # Dilate labels (use custom operation to maintain label values)
            # For each label, create a mask and dilate
            unique_labels = torch.unique(labels)
            unique_labels = unique_labels[unique_labels > 0]  # Skip background
            
            # If no foreground labels, we're done
            if len(unique_labels) == 0:
                break
            
            # Create temporary labels tensor for updating
            new_labels = labels.clone()
            
            for label in unique_labels:
                # Create binary mask for this label
                mask = (labels == label)
                
                # Dilate mask
                dilated = binary_dilation(mask, kernel_size, 1, device)
                
                # Only update where dilated mask intersects with foreground
                update_mask = dilated & foreground & (labels == 0)
                new_labels[update_mask] = label
            
            # Update labels
            labels = new_labels
            
            # Check convergence
            if torch.all(labels == prev_labels):
                break
        
        # Relabel to have consecutive IDs
        unique_labels = torch.unique(labels)
        unique_labels = unique_labels[unique_labels > 0]  # Skip background
        
        # Create mapping from old labels to new labels
        mapping = torch.zeros(torch.max(labels).item()+1, dtype=torch.int32, device=device)
        for i, label in enumerate(unique_labels):
            mapping[label] = i + 1  # Start from 1
        
        # Apply mapping
        if torch.numel(unique_labels) > 0:
            # Create output tensor
            output = torch.zeros_like(labels)
            
            # Apply mapping for each unique label
            for label in unique_labels:
                output[labels == label] = mapping[label]
            
            return output, len(unique_labels)
        else:
            return labels, 0