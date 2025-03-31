"""
Configuration module for CT2US pipeline.

This module provides configuration classes for the CT2US pipeline to make it 
more configurable and reduce hard-coding of parameters.
"""
import os
from typing import Literal, Dict, Type, Any

# Import here to avoid circular imports
import pipeline.component_classes as cc
from pipeline.base_component import BaseComponent

class PipelineConfig:
    """Configuration container for the entire CT2US pipeline."""
    
    def __init__(self, **kwargs):
        """Initialize with any kwargs that will become attributes."""
        # Check CUDA availability
        import torch
        cuda_available = torch.cuda.is_available()
        
        # Base pipeline settings - properly detect CUDA
        default_device = 'cuda' if cuda_available else 'cpu'
        self.device = kwargs.get('device', default_device)
        
        # If CUDA was requested but isn't available, fall back to CPU
        if self.device == 'cuda' and not cuda_available:
            print("CUDA requested but not available. Falling back to CPU.")
            self.device = 'cpu'
            
        # Print device being used
        print(f"Pipeline configured to use device: {self.device}")
        
        self.method = kwargs.get('method', 'total')
        self.us_method = kwargs.get('us_method', 'lotus')
        
        # Method registries - will be populated after component_classes is imported
        self.methods = {
            'segmentation': {},
            'rendering': {},
            'pointcloud': {}
        }
        
        # Component configs
        self.segmentation_config = kwargs.get('segmentation_config', {})
        self.rendering_config = kwargs.get('rendering_config', {})
        self.pointcloud_config = kwargs.get('pointcloud_config', {})
        
        # Intermediate saving settings
        self.save_intermediates = kwargs.get('save_intermediates', False)
        self.intermediate_dir = kwargs.get('intermediate_dir', './intermediates')
        
        # Update any other provided settings
        for key, value in kwargs.items():
            if key not in ['save_intermediates', 'intermediate_dir', 'device', 'method', 'us_method', 
                          'segmentation_config', 'rendering_config', 'pointcloud_config']:
                setattr(self, key, value)
        
        # Register default methods if component_classes is already imported
        self._register_default_methods()
    
    def _register_default_methods(self):
        """Register default methods after component_classes is imported."""
        # Register segmentation methods
        if hasattr(cc, 'TotalSegmentator'):
            self.register_method('segmentation', cc.TotalSegmentator)
        if hasattr(cc, 'TotalSegmentatorFast'):
            self.register_method('segmentation', cc.TotalSegmentatorFast)
        if hasattr(cc, 'PredictorSegmentator'):
            self.register_method('segmentation', cc.PredictorSegmentator)
            
        # Register rendering methods
        if hasattr(cc, 'LotusUltrasoundRenderer'):
            self.register_method('rendering', cc.LotusUltrasoundRenderer)
        if hasattr(cc, 'OptimizedLotusRenderer'):
            self.register_method('rendering', cc.OptimizedLotusRenderer)
            
        # Register pointcloud methods
        if hasattr(cc, 'PointCloudSampler'):
            self.register_method('pointcloud', cc.PointCloudSampler)
    
    def register_method(self, method_type: Literal['segmentation', 'rendering', 'pointcloud'], 
                        method_class: Type[BaseComponent]):
        """Register a new method for use in the pipeline.
        
        Args:
            method_type: Type of method ('segmentation', 'rendering', or 'pointcloud')
            method_class: Method class to register
        """
        if method_type not in self.methods:
            raise ValueError(f"Invalid method type: {method_type}")
            
        self.methods[method_type][method_class.name()] = method_class
        
    def get_method_class(self, method_type: Literal['segmentation', 'rendering', 'pointcloud'], 
                       method_name: str) -> Type[BaseComponent]:
        """Get a specific method class by type and name.
        
        Args:
            method_type: Type of method ('segmentation', 'rendering', or 'pointcloud')
            method_name: Name of the method to retrieve
            
        Returns:
            The method class
        """
        if method_type not in self.methods:
            raise ValueError(f"Invalid method type: {method_type}")
            
        if method_name not in self.methods[method_type]:
            raise ValueError(f"Method '{method_name}' not registered for type '{method_type}'")
            
        return self.methods[method_type][method_name]
    
    def instantiate_method(self, method_type: Literal['segmentation', 'rendering', 'pointcloud'], 
                         method_name: str, **kwargs) -> BaseComponent:
        """Instantiate a component from the registered method.
        
        Args:
            method_type: Type of method ('segmentation', 'rendering', or 'pointcloud')
            method_name: Name of the method to instantiate
            **kwargs: Additional parameters to pass to the constructor
            
        Returns:
            An instance of the requested component
        """
        method_class = self.get_method_class(method_type, method_name)
        
        # Merge config with kwargs for the specific method type
        merged_kwargs = {}
        if method_type == 'segmentation':
            merged_kwargs.update(self.segmentation_config)
        elif method_type == 'rendering':
            merged_kwargs.update(self.rendering_config)
        elif method_type == 'pointcloud':
            merged_kwargs.update(self.pointcloud_config)
            
        # Add device and any other kwargs
        merged_kwargs['device'] = self.device
        merged_kwargs.update(kwargs)
        
        # Components expect a single kwargs dictionary
        return method_class(merged_kwargs)
    
    def set_segmentator(self, method_name: str):
        """Set the segmentation method by name."""
        self.method = method_name
    
    def set_renderer(self, method_name: str):
        """Set the rendering method by name."""
        self.us_method = method_name
    
    def set_sampler(self, method_name: str):
        """Set the point cloud sampling method by name."""
        self.pointcloud_method = method_name
    
    def update_config(self, config_type: Literal['segmentation', 'rendering', 'pointcloud'], **kwargs):
        """Update configuration for a specific component type.
        
        Args:
            config_type: Type of configuration to update
            **kwargs: Configuration parameters to update
        """
        if config_type == 'segmentation':
            self.segmentation_config.update(kwargs)
        elif config_type == 'rendering':
            self.rendering_config.update(kwargs)
        elif config_type == 'pointcloud':
            self.pointcloud_config.update(kwargs)
        else:
            raise ValueError(f"Invalid config type: {config_type}")

class CT2USPipelineFactory:
    """Factory for creating CT2US pipeline instances with configurations."""
    
    def __init__(self):
        """Initialize the pipeline factory."""
        # Create configuration with proper device detection
        self.config = PipelineConfig(device=self.determine_device())
        
        # Register default methods
        # This is now handled by the PipelineConfig._register_default_methods()
    
    @staticmethod
    def determine_device(device='cuda', gpu_memory_threshold_gb=4):
        """
        Determine whether to use CPU or GPU for processing
        based on available GPU memory.
        
        Args:
            device: Requested device
            gpu_memory_threshold_gb: Minimum required free GPU memory in GB
            
        Returns:
            str: 'cuda' or 'cpu'
        """
        # Default to CPU for safety
        target_device = 'cpu'
        
        # Skip if user explicitly requested CPU
        if device == 'cpu':
            print("CPU explicitly requested. Using CPU for processing.")
            return 'cpu'
        
        # Check if CUDA is available
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            
            if not cuda_available:
                print("CUDA is not available on this system. Using CPU.")
                return 'cpu'
                
            # CUDA is available, check memory
            try:
                # Check all available GPUs and find the one with most free memory
                best_gpu_id = 0
                best_free_mem = 0
                
                for gpu_id in range(torch.cuda.device_count()):
                    # Get device properties
                    device_name = torch.cuda.get_device_name(gpu_id)
                    total_mem = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)  # GB
                    allocated_mem = torch.cuda.memory_allocated(gpu_id) / (1024**3)  # GB
                    free_mem = total_mem - allocated_mem
                    
                    print(f"GPU {gpu_id} ({device_name}) - Total: {total_mem:.2f}GB, Used: {allocated_mem:.2f}GB, Free: {free_mem:.2f}GB")
                    
                    if free_mem > best_free_mem:
                        best_free_mem = free_mem
                        best_gpu_id = gpu_id
                
                # Use GPU if enough memory is available
                if best_free_mem > gpu_memory_threshold_gb:
                    target_device = 'cuda'
                    # Set current device to the best GPU
                    torch.cuda.set_device(best_gpu_id)
                    print(f"Using GPU {best_gpu_id} for processing (free memory: {best_free_mem:.2f}GB)")
                    
                    # Also set environment variable for libraries that use it
                    import os
                    os.environ['CUDA_VISIBLE_DEVICES'] = str(best_gpu_id)
                else:
                    print(f"Not enough GPU memory for processing: {best_free_mem:.2f}GB available, {gpu_memory_threshold_gb:.2f}GB minimum required")
                    print("Fallback to CPU for stability")
                    
                    # Clear any allocations to free up memory
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error checking GPU memory: {e}. Using CPU for safety.")
                import traceback
                traceback.print_exc()
        except Exception as e:
            print(f"Error initializing CUDA: {e}. Using CPU for processing.")
            import traceback
            traceback.print_exc()
        
        return target_device

    def register_segmentation_method(self, method_class):
        """Register a new segmentation method.
        
        Args:
            method_class: Segmentation method class to register
        """
        self.config.register_method('segmentation', method_class)

    def register_rendering_method(self, method_class):
        """Register a new rendering method.
        
        Args:
            name: Name of the rendering method
            method_class: Rendering method class to register
        """
        self.config.register_method('rendering', method_class)
        
    def register_pointcloud_method(self, method_class):
        """Register a new point cloud method.
        
        Args:
            name: Name of the point cloud method
            method_class: Point cloud method class to register
        """
        self.config.register_method('pointcloud', method_class)

    def create_pipeline(self, config: PipelineConfig = None, **kwargs):
        """Create a CT2US pipeline with the given configuration.
        
        Args:
            config: Optional pipeline configuration
            **kwargs: Override configuration parameters
            
        Returns:
            A configured CT2US pipeline instance
        """
        # Create default config if none provided
        if config is None:
            config = self.config
            
        # Update config with any override parameters
        for key, value in kwargs.items():
            if key in ['segmentation_config', 'rendering_config', 'pointcloud_config']:
                # Update nested config dictionaries
                config_type = key.split('_')[0]
                config.update_config(config_type, **value)
            else:
                # Set attribute directly
                setattr(config, key, value)
        
        # Ensure the intermediate directory exists if needed
        if config.save_intermediates:
            os.makedirs(config.intermediate_dir, exist_ok=True)
        
        # Import here to avoid circular imports
        from pipeline.ct2us_pipeline import CT2USPipeline

        # Create segmentation component
        segmentator = config.instantiate_method('segmentation', config.method)
        
        # Create rendering component
        us_renderer = config.instantiate_method('rendering', config.us_method)
        
        # Create point cloud component (default to 'default' if not specified)
        pointcloud_method = getattr(config, 'pointcloud_method', 'default')
        pcd_sampler = config.instantiate_method('pointcloud', pointcloud_method)
        
        # Create and return the pipeline
        return CT2USPipeline(
            device_str=config.device,
            segmentation=segmentator,
            us_renderer=us_renderer,
            pcd_sampler=pcd_sampler,
            save_intermediates=config.save_intermediates,
            intermediate_dir=config.intermediate_dir
        )