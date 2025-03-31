"""
Initialize predictors for TotalSegmentator.

This module provides functions to initialize nnUNet predictors for TotalSegmentator tasks.
"""

import os
from typing import Dict, List, Union

import torch
import numpy as np

from totalsegmentator.libs import download_model_with_license_and_unpack, download_url_and_unpack
from totalsegmentator.config import get_weights_dir
from totalsegmentator.map_to_binary import commercial_models

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def initialize_predictors(device,
                         folds: list = (0,)) -> dict:
    """
    Initialize nnUNetPredictor instances for each segmentation task.

    Args:
        device (str): Device to run predictions on (device, 'cpu', 'mps').
        folds (tuple): Fold indices to use for prediction.

    Returns:
        dict: Dictionary mapping task names to their respective nnUNetPredictor instances.
    """
    # Define tasks
    tasks = [("total",
            [291, 292, 293, 294, 295],
            ["Dataset291_TotalSegmentator_part1_organs_1559subj",
            "Dataset292_TotalSegmentator_part2_vertebrae_1532subj",
            "Dataset293_TotalSegmentator_part3_cardiac_1559subj",
            "Dataset294_TotalSegmentator_part4_muscles_1559subj",
            "Dataset295_TotalSegmentator_part5_ribs_1559subj"],
            ["/v2.0.0-weights/Dataset291_TotalSegmentator_part1_organs_1559subj.zip",
            "/v2.0.0-weights/Dataset292_TotalSegmentator_part2_vertebrae_1532subj.zip",
            "/v2.0.0-weights/Dataset293_TotalSegmentator_part3_cardiac_1559subj.zip",
            "/v2.0.0-weights/Dataset294_TotalSegmentator_part4_muscles_1559subj.zip",
            "/v2.0.0-weights/Dataset295_TotalSegmentator_part5_ribs_1559subj.zip"],
            "nnUNetTrainerNoMirroring",
            False),
            ("tissue_types",
            [481],
            ["Dataset481_tissue_1559subj"],
            [],
            "nnUNetTrainer",
            True),
            ("body",
            [299],
            ["Dataset299_body_1559subj"],
            ["/v2.0.0-weights/Dataset299_body_1559subj.zip"],
            "nnUNetTrainer",
            False)]

    # Invert commercial models dictionary for lookup
    commercial_models_inv = {v: k for k, v in commercial_models.items()}
    base_url = "https://github.com/wasserth/TotalSegmentator/releases/download"

    # Get weights directory
    weights_dir = get_weights_dir()
    os.makedirs(weights_dir, exist_ok=True)

    # Dictionary to store predictors
    predictors = {}
    
    # Process each task
    for task_name, task_ids, paths, urls, trainer, with_license in tasks:
        print(f"INIT: {task_name} predictor")
        
        # Handle tasks that require a license
        if with_license:
            for i in range(len(task_ids)):
                cfg_dataset = weights_dir / paths[i] / (trainer + '__nnUNetPlans__3d_fullres') / 'dataset.json'
                if paths[i] not in os.listdir(weights_dir):
                    download_model_with_license_and_unpack(commercial_models_inv[task_ids[i]], weights_dir)

                # Initialize the predictor
                predictor = nnUNetPredictor(
                    tile_step_size=0.5,
                    use_gaussian=True,
                    use_mirroring=False,
                    perform_everything_on_device=(device != 'cpu'),
                    device=device,
                    verbose=True,
                    allow_tqdm=True
                )
                
                # Initialize from the trained model folder
                predictor.initialize_from_trained_model_folder(
                    str(weights_dir / paths[i] / (trainer + "__nnUNetPlans__3d_fullres")),
                    use_folds=folds,
                    checkpoint_name='checkpoint_final.pth'
                )

                predictors[task_ids[i]] = predictor
        
        # Handle tasks that don't require a license
        else:
            for i in range(len(urls)):
                cfg_dataset = weights_dir / paths[i] / (trainer + '__nnUNetPlans__3d_fullres') / 'dataset.json'
                if paths[i] not in os.listdir(weights_dir):
                    download_url_and_unpack(base_url + urls[i], weights_dir)

                # Initialize the predictor
                predictor = nnUNetPredictor(
                    tile_step_size=0.5,
                    use_gaussian=True,
                    use_mirroring=False,
                    perform_everything_on_device=(device != 'cpu'),
                    device=device,
                    verbose=True,
                    allow_tqdm=True
                )
                
                # Initialize from the trained model folder
                predictor.initialize_from_trained_model_folder(
                    str(weights_dir / paths[i] / (trainer + "__nnUNetPlans__3d_fullres")),
                    use_folds=folds,
                    checkpoint_name='checkpoint_final.pth'
                )
                
                predictors[task_ids[i]] = predictor

    return predictors