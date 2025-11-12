"""
Input/Output utilities for reading MRI images and contours.

This module provides functions to read medical images and their corresponding
ground truth contours from disk.
"""
from os.path import join
import numpy as np
import SimpleITK as sitk
from typing import List, Tuple, Union


def readImgAndProstate(inFolder: str, current_folder: str) -> List[sitk.Image]:
    """
    Reads the high-resolution and original transversal images and prostate contours.
    
    Args:
        inFolder: Input folder containing case directories
        current_folder: Name of the current case folder (e.g., 'Case-0001')
        
    Returns:
        List containing:
            - img_tra_original: Original resolution transversal image
            - img_tra_HR: High-resolution (0.5mm isotropic) transversal image
            - ctr_pro: Prostate contour at original resolution
            - ctr_pro_HR: Prostate contour at high resolution
            - roi_ctr_pro: ROI prostate contour
            - startROI: Starting coordinates of ROI [x, y, z]
            - sizeROI: Size of ROI [x, y, z]
            
    Raises:
        FileNotFoundError: If required files are not found
    """
    img_tra_original = sitk.ReadImage(join(inFolder, current_folder, 'img_tra.nrrd'))
    ctr_pro = sitk.ReadImage(join(inFolder, current_folder, 'ctr_pro.nrrd'))
    
    img_tra_HR = sitk.ReadImage(join(inFolder, current_folder, 'hr_tra.nrrd'))
    ctr_pro_HR = sitk.ReadImage(join(inFolder, current_folder, 'hr_ctr_pro.nrrd'))
    
    startROI = [int(x) for x in np.loadtxt(join(inFolder, current_folder, 'start_ROI.csv'))]
    sizeROI = [int(x) for x in np.loadtxt(join(inFolder, current_folder, 'size_ROI.csv'))]
    
    roi_ctr_pro = sitk.ReadImage(join(inFolder, current_folder, 'roi_ctr_pro.nrrd'))
    return [img_tra_original, img_tra_HR, ctr_pro, ctr_pro_HR, roi_ctr_pro, startROI, sizeROI]


def readROI(inFolder: str, current_folder: str, type_segmentation: str) -> List[sitk.Image]:
    """
    Reads the ROI (Region of Interest) files for different segmentation types.
    
    Args:
        inFolder: Input folder containing case directories
        current_folder: Name of the current case folder (e.g., 'Case-0001')
        type_segmentation: Type of segmentation - 'Prostate', 'PZ', or 'Lesion'
        
    Returns:
        List of ROI images:
            - For 'Prostate' or 'PZ': [roi_tra, roi_sag, roi_cor]
            - For 'Lesion': [roi_tra, roi_adc, roi_bval]
            
    Raises:
        ValueError: If type_segmentation is not recognized
        FileNotFoundError: If required files are not found
    """
    roi_img_tra = sitk.ReadImage(join(inFolder, current_folder, 'roi_tra.nrrd'))
    
    if type_segmentation == 'PZ' or type_segmentation == 'Prostate':
        roi_img_sag = sitk.ReadImage(join(inFolder, current_folder, 'roi_sag.nrrd'))
        roi_img_cor = sitk.ReadImage(join(inFolder, current_folder, 'roi_cor.nrrd'))
        return [roi_img_tra, roi_img_sag, roi_img_cor]
    
    if type_segmentation == 'Lesion':
        roi_img_bval = sitk.ReadImage(join(inFolder, current_folder, 'roi_bval.nrrd'))
        roi_img_adc = sitk.ReadImage(join(inFolder, current_folder, 'roi_adc.nrrd'))
        return [roi_img_tra, roi_img_adc, roi_img_bval]
    
    raise ValueError(f"Unknown segmentation type: {type_segmentation}. "
                    f"Must be 'Prostate', 'PZ', or 'Lesion'.")


def readPZ(inFolder: str, current_folder: str, multistream: bool, img_size: int) -> List[sitk.Image]:
    """
    Reads the peripheral zone (PZ) contours for a case.
    
    Args:
        inFolder: Input folder containing case directories
        current_folder: Name of the current case folder (e.g., 'Case-0001')
        multistream: Whether using multistream model (currently unused but kept for compatibility)
        img_size: Image size (currently unused but kept for compatibility)
        
    Returns:
        List containing:
            - ctr_pz: PZ contour at original resolution
            - ctr_pz_HR: PZ contour at high resolution
            - roi_ctr_pz: ROI PZ contour
            
    Raises:
        FileNotFoundError: If required files are not found
    """
    ctr_pz_HR = sitk.ReadImage(join(inFolder, current_folder, 'hr_ctr_pz.nrrd'))
    ctr_pz = sitk.ReadImage(join(inFolder, current_folder, 'ctr_pz.nrrd'))
    roi_ctr_pz = sitk.ReadImage(join(inFolder, current_folder, 'roi_ctr_pz.nrrd'))
    
    return [ctr_pz, ctr_pz_HR, roi_ctr_pz]
