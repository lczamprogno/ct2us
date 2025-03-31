
import torch
import torch.nn.functional as F
import numpy as np

def binary_dilation(tensor, kernel_size=3, iterations=1, device=None):
    """Dilate a binary tensor using max pooling."""
    if device is None:
        device = tensor.device if isinstance(tensor, torch.Tensor) else 'cpu'
        
    # Convert to tensor if needed
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor, device=device)
    
    # Ensure we have a batch and channel dimension
    orig_shape = tensor.shape
    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(1)
    
    # Cast to float
    tensor = (tensor > 0.5).float()
    
    # Use more robust dilation with circular kernel
    radius = kernel_size // 2
    y, x = torch.meshgrid(
        torch.arange(-radius, radius + 1, device=device),
        torch.arange(-radius, radius + 1, device=device),
        indexing='ij'
    )
    kernel = ((x ** 2 + y ** 2) <= radius ** 2).float()
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    padding = kernel_size // 2
    
    # Apply dilation
    result = tensor
    for _ in range(iterations):
        result = (F.conv2d(result, kernel, padding=padding) > 0).float()
    
    # Restore original shape
    if len(orig_shape) == 2:
        result = result.squeeze(0).squeeze(0)
    elif len(orig_shape) == 3:
        result = result.squeeze(1)
    
    return result

def binary_erosion(tensor, kernel_size=3, iterations=1, device=None):
    """Erode a binary tensor using a circular structuring element."""
    if device is None:
        device = tensor.device if isinstance(tensor, torch.Tensor) else 'cpu'
        
    # Convert to tensor if needed
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor, device=device)
    
    # Ensure we have a batch and channel dimension
    orig_shape = tensor.shape
    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(1)
    
    # Cast to float
    tensor = (tensor > 0.5).float()
    
    # Create circular kernel for more natural erosion
    radius = kernel_size // 2
    y, x = torch.meshgrid(
        torch.arange(-radius, radius + 1, device=device),
        torch.arange(-radius, radius + 1, device=device),
        indexing='ij'
    )
    kernel = ((x ** 2 + y ** 2) <= radius ** 2).float()
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    total_weight = kernel.sum().item()
    
    # Apply erosion
    result = tensor
    for _ in range(iterations):
        conv_result = F.conv2d(result, kernel, padding=radius)
        result = (conv_result >= total_weight).float()
    
    # Restore original shape
    if len(orig_shape) == 2:
        result = result.squeeze(0).squeeze(0)
    elif len(orig_shape) == 3:
        result = result.squeeze(1)
    
    return result

def binary_closing(tensor, kernel_size=3, iterations=1, device=None):
    """Apply closing (dilation followed by erosion)."""
    dilated = binary_dilation(tensor, kernel_size, iterations, device)
    return binary_erosion(dilated, kernel_size, iterations, device)

def binary_fill_holes(tensor, device=None):
    """Fill holes in a binary tensor using a more robust approach."""
    if device is None:
        device = tensor.device if isinstance(tensor, torch.Tensor) else 'cpu'
    
    # Convert to tensor if needed
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor, device='cpu')  # Start on CPU for numpy operations
    else:
        tensor = tensor.detach().cpu()  # Move to CPU for numpy operations
    
    # Convert to numpy for flood fill operations
    binary = tensor.numpy() > 0.5
    
    # Process 2D case directly
    if len(binary.shape) == 2:
        # Create a padded array for flood fill
        h, w = binary.shape
        padded = np.zeros((h+2, w+2), dtype=np.uint8)
        padded[1:-1, 1:-1] = binary.astype(np.uint8)
        
        # Create a mask to track visited pixels during flood fill
        temp = padded.copy()
        
        # Start flood fill from the boundary
        queue = [(0, 0)]
        temp[0, 0] = 2  # Mark as visited
        
        # Process the queue using a more efficient approach
        while queue:
            y, x = queue.pop(0)
            for ny, nx in [(y+1, x), (y-1, x), (y, x+1), (y, x-1)]:
                if 0 <= ny < h+2 and 0 <= nx < w+2 and temp[ny, nx] == 0:
                    temp[ny, nx] = 2  # Mark as visited
                    queue.append((ny, nx))
        
        # Pixels that weren't reached from the outside are holes
        filled = binary.copy()
        for i in range(h):
            for j in range(w):
                if temp[i+1, j+1] == 0:  # If not reached by flood fill, it's a hole
                    filled[i, j] = 1
        
        # Convert back to tensor on the specified device
        return torch.from_numpy(filled.astype(np.float32)).to(device)
    
    # Handle 3D or batched data by processing slice by slice
    original_shape = binary.shape
    result = np.zeros_like(binary)
    
    # Process each slice appropriately based on tensor dimension
    if len(original_shape) == 3:
        # 3D volume case
        for i in range(original_shape[0]):
            result[i] = binary_fill_holes(torch.from_numpy(binary[i]), device='cpu').cpu().numpy()
    elif len(original_shape) == 4:
        # Batched 3D case
        for b in range(original_shape[0]):
            for c in range(original_shape[1]):
                result[b, c] = binary_fill_holes(torch.from_numpy(binary[b, c]), device='cpu').cpu().numpy()
    
    # Convert final result back to tensor
    return torch.from_numpy(result.astype(np.float32)).to(device)

def connected_components(tensor, device=None):
    """Find connected components in a binary tensor."""
    if device is None:
        device = tensor.device if isinstance(tensor, torch.Tensor) else 'cpu'
    
    # Convert to numpy for processing
    if isinstance(tensor, torch.Tensor):
        binary = tensor.detach().cpu().numpy() > 0.5
    else:
        binary = np.asarray(tensor) > 0.5
    
    # Process with scipy
    from scipy import ndimage
    labels, num_labels = ndimage.label(binary)
    
    # Convert back to tensor
    return torch.from_numpy(labels).to(device), num_labels
