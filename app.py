try:
    # Import pipeline components
    from pipeline.dataset import CTDataset
    from pipeline.pipeline_config import CT2USPipelineFactory, PipelineConfig
except ImportError:
    from ct2us.pipeline.dataset import CTDataset
    from ct2us.pipeline.pipeline_config import CT2USPipelineFactory, PipelineConfig

global license
license = ""

# Make sure license is set in environment
import os
if license:
    os.environ["TS_LICENSE_KEY"] = license
    print(f"Set TS_LICENSE_KEY in environment")
    
    # Verify the license is set
    if "TS_LICENSE_KEY" in os.environ and os.environ["TS_LICENSE_KEY"]:
        print(f"Confirmed TS_LICENSE_KEY is set in environment")
    else:
        print(f"Warning: Failed to set TS_LICENSE_KEY in environment")
        
    # Also try to set it in totalsegmentator directly if available
    try:
        import totalsegmentator.python_api as ts
        ts.set_license_number(license)
        print(f"Set license directly in TotalSegmentator API")
    except ImportError:
        print(f"Note: TotalSegmentator not yet imported, license will be set during initialization")
else:
    print(f"Warning: No license key available to set in environment")

import sys

from pathlib import PosixPath as pthlib
from zipfile import ZipFile
import random

from math import pi
import matplotlib.pyplot as plt
import tqdm

from itertools import islice
import glob
import shutil

# Use string for path to avoid PosixPath issues with sys.path
this_folder = str(pthlib("../CT2US").resolve())

import numpy as np

sys.path.append(this_folder)
ts_cfg_path = pthlib(this_folder).joinpath(".totalsegmentator")
ts_cfg_path.mkdir(exist_ok=True, parents=True)
os.environ["TOTALSEG_HOME_DIR"] = str(ts_cfg_path)

# First check CUDA availability and print status
import torch
print(f"PyTorch version: {torch.__version__}")

try:
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Will use CPU instead.")
        # If CUDA is not available, explicitly disable it
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
except Exception as e:
    print(f"Error checking CUDA availability: {e}")
    print("This appears to be a CPU-only environment. Using CPU mode.")
    cuda_available = False
    # Set environment variables to prevent CUDA/GPU usage
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Try to import numba and CUDA components - handle failures gracefully
try:
    from numba import jit, njit
    print("Numba loaded successfully")
    
    # Only try to import CUDA components if CUDA is available
    if cuda_available:
        try:
            from numba import cuda as numba_cuda
            print("Numba CUDA loaded successfully")
        except ImportError as e:
            print(f"Error loading Numba CUDA: {e}")
except ImportError as e:
    print(f"Error loading Numba: {e}")

# Try to import cupy if CUDA is available
if cuda_available:
    try:
        import cupy as cp
        import cupyx.scipy.ndimage as cusci
        print(f"CuPy version: {cp.__version__}")
        print("CuPy and cusci loaded successfully")
    except ImportError as e:
        print(f"Error loading cupy or cusci even though CUDA is available: {e}")
        cp = None
        cusci = None
else:
    print("Not attempting to load CuPy since CUDA is not available")
    cp = None
    cusci = None

import scipy.ndimage
import scipy

from torchvision import transforms
from torch import device
from torch import uint8

from torch.utils.data import DataLoader

import gradio as gr

# Set default device based on CUDA availability
if cuda_available:
    device = torch.device("cuda", 0)
    print(f"Using device: {device}")
else:
    device = torch.device("cpu")
    print(f"Using device: {device} (CPU-only mode)")

# Set default device for PyTorch operations
try:
    torch.set_default_device(device)
except Exception as e:
    print(f"Error setting default device: {e}")
    # Fallback for older PyTorch versions
    if hasattr(torch, 'set_default_tensor_type'):
        if device.type == 'cuda':
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

# Define base directories
img_dir = pthlib(this_folder).joinpath("imgs")
label_dir = pthlib(this_folder).joinpath("labels")
us_dir = pthlib(this_folder).joinpath("us")
gen_dir = pthlib(this_folder).joinpath("gen")

# Create directories if they don't exist
os.makedirs(str(img_dir), exist_ok=True)
os.makedirs(str(label_dir), exist_ok=True)
os.makedirs(str(us_dir), exist_ok=True)
os.makedirs(str(gen_dir), exist_ok=True)

# Define tissue types mapping for the UI
TISSUE_TYPES = {
    0: "Background",
    1: "Background",
    2: "Lung",
    3: "Fat",
    4: "Vessel",
    5: "Unused",
    6: "Kidney",
    7: "Unused",
    8: "Muscle",
    9: "Background",
    10: "Unused",
    11: "Liver",
    12: "Soft Tissue",
    13: "Bone"
}

# Define base factory for the pipeline
try:
    from pipeline.component_classes import SegmentationMethod
    from pipeline.component_classes import UltrasoundRenderingMethod
    from pipeline.component_classes import OptimizedLotusRenderer
except ImportError:
    from ct2us.pipeline.component_classes import OptimizedLotusRenderer
    from ct2us.pipeline.component_classes import SegmentationMethod
    from ct2us.pipeline.component_classes import UltrasoundRenderingMethod


global _factory

_factory = CT2USPipelineFactory()

class CACTUSS(UltrasoundRenderingMethod):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.description = "CACTUSS ultrasound rendering method"
        self.tissue_types = TISSUE_TYPES

    def render(self, data):
        # Implement the rendering logic here
        raise NotImplementedError("Placeholder for CACTUSS rendering logic")

    def name():
        return "CACTUSS"

class PrelabeledBypass(SegmentationMethod):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.description = "Used to bypass segmentation and use pre-labeled data"

    def segment(self, 
                imgs: list[np.ndarray],
                properties: list[dict],
                task: str,
                resamp_thr: int):

        ret = []
        for img in imgs:
            # Convert to uint8 and return
            img = np.asarray(img.dataobj, dtype=np.uint8)
            if self.m == cp:
                img = self.m.asarray(img)
            ret.append(img)
        return ret

    def assemble(self,
                task: str,
                segs: list[np.ndarray],
                bases: list[np.ndarray],
                prev: list[np.ndarray]):
        return segs
    
    def tasks(self):
        return ["bypass"]

    def name():
        return "PrelabeledBypass"

_factory.register_rendering_method(CACTUSS)
_factory.register_segmentation_method(PrelabeledBypass)

def process_ct_images(segmentation_method, rendering_method, step_size=1, **kwargs):
    """Process CT images using the CT2US pipeline with configurable parameters.

    Args:
        ct_images: List of paths to CT images (.nii.gz files)
        step_size: Step size for slicing the volume
        segmentation_method: Segmentation method to use ("TotalSegmentator" or "TotalSegmentatorFast")
        rendering_method: Rendering method to use ("lotus")
        **kwargs: Additional configuration parameters to pass to components

    Returns:
        Tuple containing:
        - List of destination label names
        - List of ultrasound images
        - List of warped labels
        - List of viewable label images
        - Dictionary with timing information
    """
    # Initialize dataset
    local_dataset = CTDataset(
        img_dir=str(img_dir),
        resample=None,
        force_cpu=kwargs.get('force_cpu', False)  # Pass force_cpu to dataset
    )

    segmentation_config['render_interp'] = kwargs.get(kwargs['render_interp'], True)

    # Create data loader
    ct_dataloader = DataLoader(
        local_dataset,
        batch_size=1,
        collate_fn=local_dataset.collate_fn
    )

    # Organize kwargs into component-specific configs
    # Extract parameters for each component type
    segmentation_config = {}
    rendering_config = {}
    pointcloud_config = {}

    # Binary operations parameters (used by both segmentation and rendering)
    for param in ['binary_dilation_iterations', 'binary_erosion_iterations', 'density_min', 'density_max']:
        if param in kwargs:
            segmentation_config[param] = kwargs[param]
            rendering_config[param] = kwargs[param]
    
    # Segmentation-specific parameters
    for param in ['use_roi', 'force_cpu', 'fast']:
        if param in kwargs:
            segmentation_config[param] = kwargs[param]
            if param == 'force_cpu' and kwargs[param]:
                print("Using CPU for all computations")

    # Rendering-specific parameters
    for param in ['resize_size', 'crop_size']:
        if param in kwargs:
            rendering_config[param] = kwargs[param]

    # Point cloud-specific parameters (if any)
    if 'pointcloud_settings' in kwargs:
        pointcloud_config.update(kwargs['pointcloud_settings'])

    # Use string paths for intermediate directory
    intermediate_dir = kwargs.get('intermediate_dir', './intermediates')
    if hasattr(intermediate_dir, 'startswith') and not intermediate_dir.startswith('/'):
        # Convert relative path to absolute if it's a string
        intermediate_dir = os.path.join(this_folder, intermediate_dir)

    # Device selection - note that force_cpu overrides CUDA availability
    force_cpu = kwargs.get('force_cpu', False)
    device_str = 'cpu' if force_cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Ensure force_cpu is consistently set in all config dictionaries
    if force_cpu:
        segmentation_config['force_cpu'] = True
        rendering_config['force_cpu'] = True
        pointcloud_config['force_cpu'] = True
    

    # Create pipeline configuration
    _factory.config.configure(
        device=device_str,
        force_cpu=force_cpu,  # Explicitly pass the force_cpu flag to PipelineConfig
        save_intermediates=kwargs.get('save_intermediates', False),
        intermediate_dir=intermediate_dir,
        segmentation_config=segmentation_config,
        rendering_config=rendering_config,
        pointcloud_config=pointcloud_config
    )

    # Set segmentation method
    _factory.config.set_segmentator(segmentation_method)

    # Set rendering method
    _factory.config.set_renderer(rendering_method)
    
    pipeline = _factory.create_pipeline()

    # Process each batch of data
    labels = []
    us_images = []
    warped_labels = []
    viewable_labels = []
    timing_info = {}

    print("Processing data...")
    for data in tqdm.tqdm(ct_dataloader, desc="Processing batch"):
        imgs, properties, dest_labels, dest_us = data

        # Process with pipeline
        label_imgs, batch_us, batch_warped, batch_viewable, batch_timing = pipeline(
            imgs, properties, dest_labels, dest_us, step_size, True
        )

        # Store processed data
        labels.extend(label_imgs)
        us_images.extend(batch_us)
        warped_labels.extend(batch_warped)
        viewable_labels.extend(batch_viewable)
        timing_info.update(batch_timing)

    return labels, us_images, warped_labels, viewable_labels, timing_info, pipeline.pcd_sampler

todo_dir = pthlib.joinpath(pthlib(this_folder), "sample")
todo_dir.mkdir(exist_ok=True)

# Set environment for Gradio
os.environ["GRADIO_ALLOWED_PATHS"] = this_folder

def update_license(x):
    global license
    license = x

with gr.Blocks() as ct_2_us:
    # Create state objects for storing data
    files = gr.State({})
    us_list = gr.State({})
    warped_list = gr.State({})
    label_list = gr.State({})
    pcdb_list = gr.State({})
    # Store UI state to restore after processing
    ui_state = gr.State({
        "method": "TotalSegmentator",
        "use_roi": True,
        "fast_mode": False
    })

    pcd_method_obj = gr.State({})

    img_idx = gr.State(0)
    slice_idx = gr.State(0)

    with gr.Row():
        with gr.Column(scale=1):
            # Input configuration
            gr.Markdown("Input Configuration")
            ct_imgs = gr.Files(
                file_types=['.nii', '.gz'],
                type='filepath',
                label="Select CT images or preprocessed labelmaps",
                interactive=True,
                file_count='multiple'
            )

            step_size = gr.Slider(
                label="Slicing step interval",
                minimum=1,
                maximum=20,
                value=1,
                step=1,
                interactive=True
            )

            license_key = gr.Textbox(
                label="License Key",
                placeholder="Enter your totalsegmentator license",
                value=license,
                interactive=True
            )


            license_key.change(fn=update_license, inputs=[license_key], outputs=None)

            # Advanced configuration (accessed via kwargs)
            with gr.Accordion("Advanced Configuration", open=False):
                # Add ROI checkbox
                use_roi = gr.Checkbox(
                    label="Use Region of Interest (ROI)",
                    value=True,
                    info="Enables ROI-based segmentation to focus on relevant anatomy",
                    interactive=True
                )
                
                # Add Force CPU checkbox
                force_cpu = gr.Checkbox(
                    label="Force CPU Mode",
                    value=False,
                    info="Force CPU usage even if GPU is available (for testing or debugging)",
                    interactive=True
                )
                
                binary_dilation_iterations = gr.Slider(
                    label="Binary Dilation Iterations",
                    minimum=0,
                    maximum=5,
                    value=2,
                    step=1,
                    interactive=True
                )

                binary_erosion_iterations = gr.Slider(
                    label="Binary Erosion Iterations",
                    minimum=0,
                    maximum=5,
                    value=3,
                    step=1,
                    interactive=True
                )

                density_min = gr.Slider(
                    label="Density Minimum",
                    minimum=-500,
                    maximum=0,
                    value=-200,
                    step=10,
                    interactive=True
                )

                density_max = gr.Slider(
                    label="Density Maximum",
                    minimum=0,
                    maximum=500,
                    value=250,
                    step=10,
                    interactive=True
                )

                resize_size = gr.Slider(
                    label="Resize Size",
                    minimum=256,
                    maximum=512,
                    value=380,
                    step=8,
                    interactive=True
                )

                crop_size = gr.Slider(
                    label="Crop Size",
                    minimum=128,
                    maximum=384,
                    value=256,
                    step=8,
                    interactive=True
                )

                save_intermediates = gr.Checkbox(
                    label="Save Intermediate Results",
                    value=False,
                    interactive=True
                )
                
                # Add Fast mode checkbox
                fast_mode = gr.Checkbox(
                    label="Use Fast Mode (3mm)",
                    value=False,
                    info="Use faster 3mm resolution for segmentation (less accurate but quicker)",
                    interactive=True
                )

            with gr.Row():
                btn = gr.Button("Segment CT", variant="primary")
                bypass = gr.Button("Load labelmap", variant="primary")
                reset = gr.Button("Reset")

            # Define reset handler
            @gr.on([reset.click], inputs=None,
                  outputs=[files, us_list, warped_list, label_list, pcdb_list])
            def reset_all():
                # Cleanup files
                for f in glob.glob(str(label_dir / '*.nii.gz')):
                    try:
                        os.remove(f)
                    except Exception as e:
                        print(f"Error removing {f}: {e}")
                        
                for f in glob.glob(str(img_dir / '*.nii.gz')):
                    try:
                        os.remove(f)
                    except Exception as e:
                        print(f"Error removing {f}: {e}")

                for f in glob.glob(str(pthlib(this_folder) / 'intermediates' / '*.nii.gz')):
                    try:
                        os.remove(f)
                    except Exception as e:
                        print(f"Error removing {f}: {e}")
                        
                for f in glob.glob(str(gen_dir / '*.glb')):
                    try:
                        os.remove(f)
                    except Exception as e:
                        print(f"Error removing {f}: {e}")
                        
                for f in glob.glob(f"{str(us_dir)}/*"):
                    try:
                        shutil.rmtree(f, ignore_errors=True)
                    except Exception as e:
                        print(f"Error removing {f}: {e}")
                        
                try:
                    if os.path.exists(f"{this_folder}/results.zip"):
                        os.remove(f"{this_folder}/results.zip")
                except Exception as e:
                    print(f"Error removing results.zip: {e}")
                    
                return {}, {}, {}, {}, {}

            with gr.Row():
                with gr.Column():
                    # Sample selection for demo
                    sample_in = gr.Dropdown(
                        choices=[i+1 for i in range(len(glob.glob(f"{this_folder}/sample/*.nii.gz")))],
                        label='Amount of samples to randomly select',
                        info='Used for demo with no input',
                        value=1
                    )

                    # Choose pipeline method
                    available_methods = _factory.config.methods["segmentation"].keys() - [PrelabeledBypass.name()]
                    available_us = _factory.config.methods["rendering"].keys() - [OptimizedLotusRenderer.name()]

                    seg_method = gr.Radio(
                        choices=available_methods,
                        value="TotalSegmentator",
                        label="Segmentation method",
                        interactive=True
                    )

                    us_method = gr.Radio(
                        choices=available_us,
                        value="LOTUS",
                        label="US rendering method",
                        interactive=True
                    )

        with gr.Column(scale=2):
            with gr.Tab(label='Pointcloud Settings', visible=False) as pcd_tab:
                @gr.render(inputs=[step_size], triggers=[us_list.change])
                def pcd_control(step):
                    # Point cloud settings - only show if pipeline is available
                    gr.Markdown("### Point Cloud Settings")

                    # Get tissue types with voxels
                    label_counts = _pcd_method.get_label_counts()
                    if label_counts and 0 in label_counts:
                        available_labels = sorted(label_counts[0].keys())

                        # Create sliders for each tissue type
                        with gr.Row():
                            with gr.Column():
                                pcd_sliders = []
                                for i, label in enumerate(available_labels):
                                    if label in TISSUE_TYPES and TISSUE_TYPES[label] != "Unused":
                                        # Get current point count
                                        if i < len(_pcd_method.points_per_label):
                                            current_value = _pcd_method.points_per_label[i]
                                        else:
                                            current_value = 0

                                        # Calculate max points
                                        max_points = min(label_counts[0].get(label, 0), 400000)

                                        # Skip labels with no points
                                        if max_points == 0:
                                            continue

                                        # Create slider
                                        slider = gr.Slider(
                                            label=f"{TISSUE_TYPES[label]} Points",
                                            minimum=0,
                                            maximum=max_points,
                                            value=current_value,
                                            step=1000,
                                            interactive=True
                                        )
                                        pcd_sliders.append((label, slider))

                                # Create update button
                                update_pcd = gr.Button("Resample Pointcloud")

                                # Handle updates
                                def update_point_cloud(x, y, *slider_values):
                                    # Get current points per label
                                    new_counts = _pcd_method.points_per_label.copy()

                                    # Update with slider values
                                    for (label, _), value in zip(pcd_sliders, slider_values):
                                        if label < len(new_counts):
                                            new_counts[label] = int(value)

                                    # Update point cloud
                                    _pcd_method.update_points_per_label(new_counts)

                                    # Re-export
                                    pcdb_new = _pcd_method.sample(x)

                                    try:
                                        # Make sure to use a safe value for the slice index
                                        safe_slice = min(int(y * step), pcdb_new[2][2]-1) if len(pcdb_new) >= 3 and hasattr(pcdb_new[2], "__len__") and len(pcdb_new[2]) >= 3 else 0
                                        _pcd_method.add_axis_pcd(pcdb_new, safe_slice).export(str(gen_dir / "current_pcd.glb"))
                                    except Exception as e:
                                        print(f"Error creating axis point cloud: {e}")
                                        # Create a fallback point cloud if there's an error
                                        import trimesh as tri
                                        fallback_pcd = tri.PointCloud([[0, 0, 0]], colors=[[255, 255, 255, 255]])
                                        fallback_pcd.export(str(gen_dir / "current_pcd.glb"))

                                    # Return the path to the new point cloud
                                    return pcdb_new

                                # Connect button to handler
                                if pcd_sliders:
                                    update_pcd.click(
                                        fn=update_point_cloud,
                                        inputs=[img_idx, slice_idx] + [s[1] for s in pcd_sliders],
                                        outputs=pcdb_list
                                    )


            with gr.Tab(label='Preview'):
                note = gr.Markdown(value="Generate US images first through the input tab")

                # Dynamic UI based on available data
                @gr.render(inputs=[files, us_list, warped_list, label_list, step_size],
                         triggers=[us_list.change])
                def dynamic(fl, us, warped, ll, step):
                    with gr.Column():
                        if len(us) > 0:
                            # Image selection
                            dropdown = gr.Dropdown(
                                choices=[(f, n) for n, f in fl.items()],
                                label='Select image to preview',
                                value=0
                            )

                            # Slice selection
                            slider = gr.Slider(
                                minimum=0,
                                maximum=len(warped[0]) - 1,
                                step=step,
                                label='Slice selection',
                                value=0
                            )

                            # Identity function for state updates
                            iden = lambda x: x

                            slider.release(fn=iden, inputs=[slider], outputs=[slice_idx])
                            dropdown.select(fn=iden, inputs=[dropdown], outputs=[img_idx])

                            # Results display
                            with gr.Column():
                                # Top row: US and label images
                                with gr.Row():
                                    base = gr.Image(
                                        label='US slice',
                                        value=np.asarray(us[img_idx.value][slice_idx.value], dtype=np.float32) if len(us) > 0 and len(us[0]) > 0 else np.zeros((256, 256), dtype=np.float32),
                                        height=300
                                    )

                                    label_preview = gr.Image(
                                        label='Label slice',
                                        value=ll[img_idx.value][slice_idx.value] if len(ll) > 0 else None,
                                        type='pil',
                                        height=300
                                    )

                                # Bottom row: Annotation and 3D view
                                with gr.Row():
                                    comp = gr.AnnotatedImage(
                                        value=(us[img_idx.value][slice_idx.value], warped[img_idx.value][slice_idx.value]),
                                        height=300
                                    )

                                    volume_preview = gr.Model3D(
                                        clear_color=(0, 0, 0, 1),
                                        label="Label map view",
                                        value=str(gen_dir / "current_pcd.glb") if os.path.exists(str(gen_dir / "current_pcd.glb")) else None,
                                        height=300
                                    )

                                pcdb_list.change(fn=lambda : str(gen_dir / "current_pcd.glb"), outputs=volume_preview)

                                # Update function for image/slice selection
                                def route(x, y, pcdb):
                                        new_y = y if y < len(us[x]) else 0

                                        b = us[x][new_y]
                                        w = warped[x][new_y]
                                        l = ll[x][new_y]

                                        # adjust current slice highlighted
                                        pcdb = _pcd_method.sample(x)

                                        # Make sure to use a safe value for the slice index
                                        safe_slice = min(int(y * step), pcdb[2][2]-1) if len(pcdb) >= 3 and hasattr(pcdb[2], "__len__") and len(pcdb[2]) >= 3 else 0
                                        _pcd_method.add_axis_pcd(pcdb, safe_slice).export(str(gen_dir / "current_pcd.glb"))

                                        # Calculate new slider value and ensure it's in range
                                        new_slider = gr.Slider(
                                            minimum=0,
                                            maximum=len(warped[x]) - 1,
                                            step=step,
                                            label='Slice selection',
                                            value=new_y
                                        )

                                        # Return updated UI components
                                        return (b, w), b, l, str(gen_dir / "current_pcd.glb"), new_y, new_slider, pcdb

                                def route_y(x, y, pcdb):
                                        b = us[x][y]
                                        w = warped[x][y]
                                        l = ll[x][y]

                                        # adjust current slice highlighted
                                        try:
                                            # Make sure to use a safe value for the slice index
                                            safe_slice = min(int(y * step), pcdb[2][2]-1) if len(pcdb) >= 3 and hasattr(pcdb[2], "__len__") and len(pcdb[2]) >= 3 else 0
                                            _pcd_method.add_axis_pcd(pcdb, safe_slice).export(str(gen_dir / "current_pcd.glb"))
                                        except Exception as e:
                                            print(f"Error creating axis point cloud: {e}")
                                            # Create a fallback point cloud if there's an error
                                            import trimesh as tri
                                            fallback_pcd = tri.PointCloud([[0, 0, 0]], colors=[[255, 255, 255, 255]])
                                            fallback_pcd.export(str(gen_dir / "current_pcd.glb"))

                                        # Return updated UI components
                                        return (b, w), b, l, str(gen_dir / "current_pcd.glb")

                                # Connect route function to UI events
                                gr.on(
                                    triggers=[img_idx.change],
                                    fn=route,
                                    inputs=[img_idx, slice_idx, pcdb_list],
                                    outputs=[comp, base, label_preview, volume_preview, slice_idx, slider, pcdb_list]
                                )
                                gr.on(
                                    triggers=[slice_idx.change],
                                    fn=route_y,
                                    inputs=[img_idx, slice_idx, pcdb_list],
                                    outputs=[comp, base, label_preview, volume_preview]
                                )

            with gr.Tab(label='Download'):
                download = gr.DownloadButton(label="", visible=False)

                # Dynamic UI for download options
                @gr.render(inputs=[files, us_list, warped_list, label_list, pcdb_list, step_size],
                         triggers=[us_list.change])
                def dynamic(fl, us, warped, ll, pcdb, step):
                    # Create download button
                    descr = gr.Markdown(label="This can be used to adjust contents of results.zip")
                    configs = gr.CheckboxGroup(
                        choices=["Save labels", "Save US images", "Save intermediates"],
                        value=["Save labels", "Save US images", "Save intermediates"],
                        label="Options",
                        interactive=True
                    )

                    intermed_dir = pthlib(this_folder) / 'intermediates'

                    rezip = gr.Button("Reassemble results.zip")

                    @gr.on(rezip.click, inputs=[configs], outputs=[download, descr])
                    def rezip_files(save_configs):
                        dest = pthlib(this_folder) / 'zip'
                        os.mkdir(dest)
                        files = glob.glob(f"{this_folder}/zip/*")
                        for f in files:
                            os.remove(f)

                        if "Save labels" in save_configs:
                            shutil.copytree(label_dir, dest / "label", dirs_exist_ok=True)

                        if "Save US images" in save_configs:
                            shutil.copytree(us_dir, dest / "us", dirs_exist_ok=True)

                        if "Save intermediates" in save_configs:
                            shutil.copytree(intermed_dir, dest / "intermediate", dirs_exist_ok=True)
                        
                        # Clean up if requested
                        if "Clean up after export" in save_configs:
                            for f in glob.glob(f"{str(us_dir)}/*"):
                                shutil.rmtree(f, ignore_errors=True)
                        
                        shutil.make_archive(f"{this_folder}/results", 'zip', dest)

                        return f"{this_folder}/results.zip", "Results have been rezipped"

                # Function to hide pointcloud tab during processing
                def hide_pcd_tab():
                    return gr.Tab(label="Pointcloud Settings", visible=False)

                def start(ct, step, method, method_us, fl_s, us_s, warped_s, ll_s, pcdb_s, nr_samples,
                         use_roi_enabled, force_cpu_enabled, 
                         binary_dilation_iters, binary_erosion_iters, 
                         dens_min, dens_max, resize, crop, save_int, fast_enabled, ui_state_val,
                         progress=gr.Progress(track_tqdm=True)):
                    # First ensure the working directory is completely clean
                    try:
                        # Clean img_dir first to avoid duplicates
                        for f in glob.glob(str(img_dir / '*.nii.gz')):
                            try:
                                os.remove(f)
                                print(f"Cleaned up existing file: {f}")
                            except Exception as e:
                                print(f"Error removing {f}: {e}")
                    except Exception as e:
                        print(f"Error during cleanup: {e}")
                        
                    # Sample if no input
                    if not ct:
                        ct = glob.glob(f"{this_folder}/sample/*.nii.gz")
                        ct = [f for f in ct]

                    # Random sample if needed
                    if len(ct) > nr_samples:
                        ct = random.sample(ct, k=nr_samples)

                    # Copy files to working directory - with unique names to avoid duplicates
                    already_copied = set()
                    for f in ct:
                        try:
                            # Get just the filename
                            base_name = os.path.basename(f)
                            
                            # Skip if we've already copied this filename
                            if base_name in already_copied:
                                print(f"Skipping duplicate file: {base_name}")
                                continue
                                
                            # Copy the file
                            dest_path = str(img_dir / base_name)
                            shutil.copyfile(f, dest_path)
                            already_copied.add(base_name)
                            print(f"Copied {base_name} to working directory")
                        except Exception as e:
                            print(f"Error copying file {f}: {e}")
                    
                    # Auto-select the 3mm fast TotalSegmentator with ROI when CPU needs to be used
                    # This happens when:
                    # 1. force_cpu is enabled (explicit user choice)
                    # 2. CUDA is not available on the system
                    original_method = method
                    original_roi = use_roi_enabled
                    original_fast = fast_enabled
                    
                    need_cpu_mode = force_cpu_enabled or not torch.cuda.is_available()
                    auto_adjusted = False
                    
                    if need_cpu_mode and not force_cpu_enabled:
                        print("CPU-only environment detected. Auto-adjusting configuration for optimal performance.")
                        auto_adjusted = True
                    
                    # If CPU needs to be used (and it's not an explicit user choice with force_cpu), 
                    # auto-select optimized settings
                    if need_cpu_mode and method != "TotalSegmentator 3mm":
                        if not force_cpu_enabled:
                            # Only auto-adjust if not explicitly forced to CPU by user
                            method = "TotalSegmentator 3mm"
                            use_roi_enabled = True
                            fast_enabled = True
                            print(f"Auto-selected 3mm fast TotalSegmentator with ROI for CPU processing")
                            auto_adjusted = True
                        else:
                            # If user forced CPU but didn't select optimal settings, just suggest it
                            if not (fast_enabled and use_roi_enabled and method == "TotalSegmentator 3mm"):
                                print("TIP: For optimal CPU performance, consider using 'TotalSegmentator 3mm' with ROI and FAST mode enabled.")

                    # Display configuration settings
                    if force_cpu_enabled:
                        print("Force CPU mode enabled")
                    
                    if use_roi_enabled:
                        print("ROI feature enabled")
                        
                    if fast_enabled:
                        print("FAST mode (3mm) enabled")

                    # Organize parameters into component-specific configs
                    # Common parameters for different components
                    segmentation_config = {
                        'binary_dilation_iterations': binary_dilation_iters,
                        'binary_erosion_iterations': binary_erosion_iters,
                        'density_min': dens_min,
                        'density_max': dens_max,
                        'use_roi': use_roi_enabled,
                        'force_cpu': force_cpu_enabled,
                        'fast': fast_enabled
                    }

                    rendering_config = {
                        'binary_dilation_iterations': binary_dilation_iters,
                        'binary_erosion_iterations': binary_erosion_iters,
                        'density_min': dens_min,
                        'density_max': dens_max,
                        'resize_size': resize,
                        'crop_size': crop,
                        'force_cpu': force_cpu_enabled
                    }

                    # Process images using the pipeline with all configuration parameters
                    labels, us_images, warped_labels, viewable_labels, timing_info, sampler = process_ct_images(
                        ct_images=ct,
                        step_size=step,
                        segmentation_method=method,
                        rendering_method=method_us,
                        save_intermediates=save_int,
                        force_cpu=force_cpu_enabled,
                        segmentation_config=segmentation_config,
                        rendering_config=rendering_config
                    )

                    # Get point cloud sampler for later adjustment
                    global _pcd_method
                    _pcd_method = sampler

                    # Update state
                    fl_s.update(enumerate(labels))
                    us_s.update(enumerate(us_images))
                    ll_s.update(enumerate(viewable_labels))
                    warped_s.update(enumerate(warped_labels))

                    # Sample point clouds and save initial 3D view
                    pcdb_s = _pcd_method.sample(0)

                    try:
                        # Create initial view with slice 0
                        _pcd_method.add_axis_pcd(pcdb_s, 0).export(str(gen_dir / "current_pcd.glb"))
                    except Exception as e:
                        print(f"Error creating initial point cloud: {e}")
                        # Create a fallback point cloud if there's an error
                        import trimesh as tri
                        fallback_pcd = tri.PointCloud([[0, 0, 0]], colors=[[255, 255, 255, 255]])
                        fallback_pcd.export(str(gen_dir / "current_pcd.glb"))
                    
                    # Create message about any auto-adjustments that were made
                    completion_message = "Processing complete!"
                    if auto_adjusted:
                        completion_message += "\nSettings were auto-adjusted for optimal CPU performance."
                    
                    # Store original vs auto-adjusted settings for UI restoration
                    auto_adjust_info = {
                        "method": {"original": original_method, "adjusted": method},
                        "use_roi": {"original": original_roi, "adjusted": use_roi_enabled},
                        "fast": {"original": original_fast, "adjusted": fast_enabled},
                        "auto_adjusted": auto_adjusted
                    }

                    dest = pthlib(this_folder) / 'zip'
                    shutil.copytree(label_dir, dest / "label", dirs_exist_ok=True)
                    shutil.copytree(us_dir, dest / "us", dirs_exist_ok=True)
                        
                    shutil.make_archive(f"{this_folder}/results", 'zip', dest)

                    # Return downloadable zip, updated states, and UI components to restore
                    return (
                        gr.DownloadButton(label="Download results as zip", visible=True, value=f"{this_folder}/results.zip"),
                        fl_s,
                        us_s,
                        warped_s,
                        ll_s,
                        pcdb_s,
                        gr.Markdown(value=completion_message, height=30),
                        auto_adjust_info
                    )
                
                def bypass_start(label_list, step, method_us, fl_s, us_s, warped_s, ll_s, pcdb_s, nr_samples,
                         use_roi_enabled, force_cpu_enabled, 
                         binary_dilation_iters, binary_erosion_iters, 
                         dens_min, dens_max, resize, crop, save_int, fast_enabled, ui_state_val,
                         progress=gr.Progress(track_tqdm=True)):
                    # First ensure the working directory is completely clean
                    try:
                        # Clean img_dir first to avoid duplicates
                        for f in glob.glob(str(img_dir / '*.nii.gz')):
                            try:
                                os.remove(f)
                                print(f"Cleaned up existing file: {f}")
                            except Exception as e:
                                print(f"Error removing {f}: {e}")
                    except Exception as e:
                        print(f"Error during cleanup: {e}")
                        
                    # Sample if no input
                    if not label_list:
                        label_list = glob.glob(f"{this_folder}/sample/prelabeled_*.nii.gz")
                        label_list = [f for f in label_list]

                    # Random sample if needed
                    if len(label_list) > nr_samples:
                        label_list = random.sample(label_list, k=nr_samples)

                    # Copy files to working directory - with unique names to avoid duplicates
                    already_copied = set()
                    for f in label_list:
                        try:
                            # Get just the filename
                            base_name = os.path.basename(f)
                            
                            # Skip if we've already copied this filename
                            if base_name in already_copied:
                                print(f"Skipping duplicate file: {base_name}")
                                continue
                                
                            # Copy the file
                            dest_path = str(img_dir / base_name)
                            shutil.copyfile(f, dest_path)
                            already_copied.add(base_name)
                            print(f"Copied {base_name} to working directory")
                        except Exception as e:
                            print(f"Error copying file {f}: {e}")
                    
                    # Auto-select the 3mm fast TotalSegmentator with ROI when CPU needs to be used
                    # This happens when:
                    # 1. force_cpu is enabled (explicit user choice)
                    # 2. CUDA is not available on the system
                    original_method = "PrelabeledBypass"
                    original_roi = use_roi_enabled
                    original_fast = fast_enabled
                    
                    need_cpu_mode = force_cpu_enabled or not torch.cuda.is_available()
                    auto_adjusted = False
                    
                    if need_cpu_mode and not force_cpu_enabled:
                        print("CPU-only environment detected. Auto-adjusting configuration for optimal performance.")
                        auto_adjusted = True
                    
                    # If CPU needs to be used (and it's not an explicit user choice with force_cpu), 
                    # auto-select optimized settings
                    if need_cpu_mode:
                        if not force_cpu_enabled:
                            # Only auto-adjust if not explicitly forced to CPU by user
                            method = ""
                            use_roi_enabled = True
                            fast_enabled = True
                            print(f"Auto-selected 'PrelabeledBypass' for processing")
                            auto_adjusted = True
                        else:
                            # If user forced CPU but didn't select optimal settings, just suggest it
                            if not (fast_enabled and use_roi_enabled and method == "TotalSegmentator 3mm"):
                                print("TIP: For optimal CPU performance, consider using 'TotalSegmentator 3mm' with ROI and FAST mode enabled.")

                    # Display configuration settings
                    if force_cpu_enabled:
                        print("Force CPU mode enabled")
                    
                    if use_roi_enabled:
                        print("ROI feature enabled")
                        
                    if fast_enabled:
                        print("FAST mode (3mm) enabled")

                    # Organize parameters into component-specific configs
                    # Common parameters for different components
                    segmentation_config = {
                        'binary_dilation_iterations': binary_dilation_iters,
                        'binary_erosion_iterations': binary_erosion_iters,
                        'density_min': dens_min,
                        'density_max': dens_max,
                        'use_roi': use_roi_enabled,
                        'force_cpu': force_cpu_enabled,
                        'fast': fast_enabled
                    }

                    rendering_config = {
                        'binary_dilation_iterations': binary_dilation_iters,
                        'binary_erosion_iterations': binary_erosion_iters,
                        'density_min': dens_min,
                        'density_max': dens_max,
                        'resize_size': resize,
                        'crop_size': crop,
                        'force_cpu': force_cpu_enabled
                    }

                    # Process images using the pipeline with all configuration parameters
                    labels, us_images, warped_labels, viewable_labels, timing_info, sampler = process_ct_images(
                        ct_images=label_list,
                        step_size=step,
                        segmentation_method="PrelabeledBypass",
                        rendering_method=method_us,
                        save_intermediates=save_int,
                        force_cpu=force_cpu_enabled,
                        segmentation_config=segmentation_config,
                        rendering_config=rendering_config
                    )

                    # Get point cloud sampler for later adjustment
                    global _pcd_method
                    _pcd_method = sampler

                    # Update state
                    fl_s.update(enumerate(labels))
                    us_s.update(enumerate(us_images))
                    ll_s.update(enumerate(viewable_labels))
                    warped_s.update(enumerate(warped_labels))

                    # Sample point clouds and save initial 3D view
                    pcdb_s = _pcd_method.sample(0)

                    try:
                        # Create initial view with slice 0
                        _pcd_method.add_axis_pcd(pcdb_s, 0).export(str(gen_dir / "current_pcd.glb"))
                    except Exception as e:
                        print(f"Error creating initial point cloud: {e}")
                        # Create a fallback point cloud if there's an error
                        import trimesh as tri
                        fallback_pcd = tri.PointCloud([[0, 0, 0]], colors=[[255, 255, 255, 255]])
                        fallback_pcd.export(str(gen_dir / "current_pcd.glb"))
                    
                    # Create message about any auto-adjustments that were made
                    completion_message = "Processing complete!"
                    if auto_adjusted:
                        completion_message += "\nSettings were auto-adjusted for optimal CPU performance."
                    
                    # Store original vs auto-adjusted settings for UI restoration
                    auto_adjust_info = {
                        "method": {"original": original_method, "adjusted": 'PrelabeledBypass'},
                        "use_roi": {"original": original_roi, "adjusted": use_roi_enabled},
                        "fast": {"original": original_fast, "adjusted": fast_enabled},
                        "auto_adjusted": auto_adjusted
                    }

                    dest = pthlib(this_folder) / 'zip'
                    shutil.copytree(label_dir, dest / "label", dirs_exist_ok=True)
                    shutil.copytree(us_dir, dest / "us", dirs_exist_ok=True)
                        
                    shutil.make_archive(f"{this_folder}/results", 'zip', dest)

                    # Return downloadable zip, updated states, and UI components to restore
                    return (
                        gr.DownloadButton(label="Download results as zip", visible=True, value=f"{this_folder}/results.zip"),
                        fl_s,
                        us_s,
                        warped_s,
                        ll_s,
                        pcdb_s,
                        gr.Markdown(value=completion_message, height=30),
                        auto_adjust_info
                    )

                def finalize(x):
                    # Make the Pointcloud settings tab visible while preserving the note visibility
                    # Do not modify the note component - return None to keep the current state
                    return gr.Markdown(label="", height=0, visible=False), gr.Tab(label="Pointcloud Settings", visible=True)

                # Store UI state before processing
                def save_ui_state(method, use_roi_val, fast_mode_val):
                    return {"method": method, "use_roi": use_roi_val, "fast_mode": fast_mode_val}
                
                # Restores UI to original settings after processing (except configs)
                def restore_ui(ui_state_val, seg_method_val, use_roi_val, fast_mode_val):
                    # Only restore if auto-adjustment was made
                    if ui_state_val.get("auto_adjusted", False):
                        return (
                            ui_state_val["method"]["original"], 
                            ui_state_val["use_roi"]["original"], 
                            ui_state_val["fast"]["original"]
                        )
                    # Otherwise keep current values
                    return seg_method_val, use_roi_val, fast_mode_val

                # Connect generate button
                btn.click(
                    # Store UI state before processing
                    fn=save_ui_state,
                    inputs=[seg_method, use_roi, fast_mode],
                    outputs=[ui_state]
                ).success(
                    fn=reset_all,
                    inputs=None,
                    outputs=[files, us_list, warped_list, label_list, pcdb_list]
                ).success(
                    # Hide pointcloud tab at start of processing
                    fn=hide_pcd_tab,
                    inputs=None,
                    outputs=[pcd_tab]
                ).success(
                    fn=lambda x: gr.Markdown(label="Status", value="Processing...", height=80),
                    inputs=btn,
                    outputs=note
                ).success(
                    fn=start,
                    inputs=[
                        ct_imgs, step_size, seg_method, us_method, files, us_list, warped_list,
                        label_list, pcdb_list, sample_in, use_roi, force_cpu, 
                        binary_dilation_iterations, binary_erosion_iterations, density_min, 
                        density_max, resize_size, crop_size, save_intermediates, 
                        fast_mode, ui_state
                    ],
                    outputs=[download, files, us_list, warped_list, label_list, pcdb_list, note, ui_state]
                ).success(
                    fn=finalize,
                    inputs=btn,
                    outputs=[note, pcd_tab]
                ).success(
                    # Restore UI to original state (if auto-adjusted)
                    fn=restore_ui,
                    inputs=[ui_state, seg_method, use_roi, fast_mode],
                    outputs=[seg_method, use_roi, fast_mode]
                )

                                # Connect generate button
                bypass.click(
                    # Store UI state before processing
                    fn=save_ui_state,
                    inputs=[seg_method, use_roi, fast_mode],
                    outputs=[ui_state]
                ).success(
                    fn=reset_all,
                    inputs=None,
                    outputs=[files, us_list, warped_list, label_list, pcdb_list]
                ).success(
                    # Hide pointcloud tab at start of processing
                    fn=hide_pcd_tab,
                    inputs=None,
                    outputs=[pcd_tab]
                ).success(
                    fn=lambda x: gr.Markdown(label="Status", value="Processing...", height=80),
                    inputs=bypass,
                    outputs=note
                ).success(
                    fn=bypass_start,
                    inputs=[
                        ct_imgs, step_size, us_method, files, us_list, warped_list,
                        label_list, pcdb_list, sample_in, use_roi, force_cpu, 
                        binary_dilation_iterations, binary_erosion_iterations, density_min, 
                        density_max, resize_size, crop_size, save_intermediates, 
                        fast_mode, ui_state
                    ],
                    outputs=[download, files, us_list, warped_list, label_list, pcdb_list, note, ui_state]
                ).success(
                    fn=finalize,
                    inputs=bypass,
                    outputs=[note, pcd_tab]
                ).success(
                    # Restore UI to original state (if auto-adjusted)
                    fn=restore_ui,
                    inputs=[ui_state, seg_method, use_roi, fast_mode],
                    outputs=[seg_method, use_roi, fast_mode]
                )

# Launch the Gradio app
ct_2_us.launch(debug=True, share=True)