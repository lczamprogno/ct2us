"""
Dataset handling for CT2US pipeline.

This module contains dataset classes for loading and preprocessing CT volumes.
"""

import glob
from pathlib import PosixPath as pthlib
import numpy as np
import torch
from torch.utils.data import Dataset
from nibabel import nifti1
from batchgenerators.utilities.file_and_folder_operations import join

try:
    import cupy as cp
except ImportError:
    pass

class CTDataset(Dataset):
    # Dataset class for loading CT volumes from directories.
    
    def __init__(self, img_dir: str, method: str = "old", annotations_file: str = None, resample: float = 1.5, force_cpu: bool = False):
        """
        Initialize the dataset.
        
        Args:
            img_dir: Directory containing CT images
            method: Pipeline method to use ('old', 'new', 'predictor')
            annotations_file: Optional path to annotations file (CSV)
            resample: Resampling factor
            force_cpu: Force CPU usage even if CUDA is available
        """
        # If force_cpu is True, always use CPU regardless of CUDA availability
        # Set device based on force_cpu flag
        self.device = torch.device("cpu") if force_cpu else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
            
        self.method = method
        
        # Set up array library based on device
        if self.device.type == "cuda":
            try:
                self.m = cp
            except NameError:
                self.m = np
        else:
            self.m = np

        # Set up collate function
        self.collate_fn = self.collate_list

        # Convert dir path if needed
        if not isinstance(img_dir, pthlib):
            img_dir = pthlib(img_dir)

        self.resample = resample
        self.img_dir = img_dir
        self.annotations_file = annotations_file
        
        # Set up loading function based on method
        self.load = self.load_bases
        self.resampler = None  # Not using resampling by default

        # Load image paths
        if self.annotations_file is not None:
            # TODO: Load paths from CSV file
            self.img_paths = []
        else:
            self.img_paths = glob.glob(f"{str(self.img_dir)}/*.nii.gz")
            self.img_paths = [(
                pth,
                pth.replace(".nii.gz", "_label.nii.gz").replace("/imgs/", "/labels/"),
                pth.replace(".nii.gz", "_us").replace("/imgs/", "/us/")
            ) for pth in self.img_paths]

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.img_paths)

    def load_bases(self, pths: str) -> tuple:
        """
        Load a CT image and its properties.
        
        Args:
            pths: Path to the image
            
        Returns:
            Tuple containing the image data and properties
        """
        # Try to load using SITK and SimpleITKIO
        try:
            from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
            img, prop = SimpleITKIO().read_images(image_fnames=[join(pths)])
            return (img, prop)
        except ImportError:
            # Fallback to nibabel
            print("SimpleITKIO not available, using nibabel")
            img = nifti1.load(pths)
            return (np.asarray(img.dataobj), {"spacing": img.header.get_zooms()})

    def collate_tensor(self, data):
        """Collate function for tensor-based methods."""
        imgs, properties, dest_labels, dest_us = zip(*data)
        return torch.from_numpy(np.stack(imgs)), properties, dest_labels, dest_us

    def collate_list(self, data):
        """Collate function for list-based methods."""
        imgs, properties, dest_labels, dest_us = zip(*data)
        return list(imgs), list(properties), list(dest_labels), list(dest_us)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple containing (image, properties, label path, US path)
        """
        todo = self.img_paths[idx]
        img, prop = self.load(todo[0])
        dest_label = todo[1]
        dest_us = todo[2]

        # Format the return value based on method
        if self.method == "predictor":
            ret = (img[0][None, ...], prop, dest_label, dest_us)
        elif self.method == "new":
            # Create affine matrix
            affine = np.eye(4)
            affine[:3, :3] = np.array(prop["sitk_stuff"]["direction"]).reshape(3, 3) * prop["sitk_stuff"]["spacing"]
            affine[:3, 3] = prop["sitk_stuff"]["origin"]
            
            # Create Nifti1Image
            ret = (nifti1.Nifti1Image(img[0].transpose(2, 1, 0), affine=affine), prop, dest_label, dest_us)
        elif self.method == "old":
            # Create affine matrix
            affine = np.eye(4)
            affine[:3, :3] = np.array(prop["sitk_stuff"]["direction"]).reshape(3, 3) * prop["sitk_stuff"]["spacing"]
            affine[:3, 3] = prop["sitk_stuff"]["origin"]
            
            # Create Nifti1Image
            ret = (nifti1.Nifti1Image(img[0].transpose(2, 1, 0), affine=affine), prop, dest_label, dest_us)

        return ret