"""
Script to perform segmentation on new cases without ground truth data.
For each case folder, it reads cor, sag, and tra images, processes them,
and uses the CNN U-Net model to make predictions.
"""
import preproc.utils as utils
import visualization.utilsviz as utilsviz
import models.models as models
import os
from os.path import join
import SimpleITK as sitk
import time
from typing import Dict, Any, List


def readOriginalImages(inFolder: str, current_folder: str) -> List[sitk.Image]:
    """
    Reads the original coronal, sagittal, and transversal images from a case folder.
    
    Args:
        inFolder: Input folder containing case directories
        current_folder: Name of the current case folder (e.g., 'Case-0001')
        
    Returns:
        List of three SimpleITK images: [img_tra, img_cor, img_sag]
        
    Raises:
        FileNotFoundError: If any of the required image files are not found
        RuntimeError: If image reading fails
    """
    case_path = join(inFolder, current_folder)
    
    # Check if case folder exists
    if not os.path.exists(case_path):
        raise FileNotFoundError(f"Case folder not found: {case_path}")
    
    # Define required files
    required_files = {
        'tra': join(case_path, 'img_tra.nrrd'),
        'cor': join(case_path, 'img_cor.nrrd'),
        'sag': join(case_path, 'img_sag.nrrd')
    }
    
    # Check if all files exist
    missing_files = [name for name, path in required_files.items() if not os.path.exists(path)]
    if missing_files:
        raise FileNotFoundError(
            f"Missing required image files in {case_path}: {', '.join(missing_files)}"
        )
    
    # Read images with error handling
    try:
        img_tra = sitk.ReadImage(required_files['tra'])
        img_cor = sitk.ReadImage(required_files['cor'])
        img_sag = sitk.ReadImage(required_files['sag'])
    except Exception as e:
        raise RuntimeError(f"Failed to read images from {case_path}: {str(e)}")
    
    return [img_tra, img_cor, img_sag]


def makePrediction(model: Any, img_size: int, roi_img1: sitk.Image, 
                   roi_img2: sitk.Image, roi_img3: sitk.Image) -> Any:
    """
    Makes a prediction using the neural network model.
    
    Args:
        model: The loaded TensorFlow/Keras model
        img_size: Size of the input images (typically 168)
        roi_img1: ROI image 1 (transversal)
        roi_img2: ROI image 2 (sagittal)
        roi_img3: ROI image 3 (coronal)
        
    Returns:
        Model prediction output
    """
    input_array = utils.createInputArray(True, img_size, roi_img1, roi_img2, roi_img3)
    output_NN = model.predict(input_array, verbose=1)
    return output_NN


def processNewCase(inFolder: str, outputDirectory: str, outputImages: str, 
                   all_params: Dict[str, Any], current_folder: str, 
                   img_size: int, model: Any, threshold: float, 
                   save_segmentations: bool = True) -> None:
    """
    Processes a single new case: reads images, preprocesses them, makes prediction, and saves results.
    
    Args:
        inFolder: Input folder where all the cases are
        outputDirectory: Where to save the prediction as nrrd file
        outputImages: Where to save visualization images
        all_params: Parameters from the user configuration
        current_folder: Current case folder name
        img_size: Image size for the model (typically 168)
        model: TensorFlow/Keras model
        threshold: Threshold to binarize the predictions
        save_segmentations: Whether to save the segmentation files
    """
    print(f'Processing case: {current_folder}')
    
    # Step 1: Read original cor, sag, tra images
    print('Reading original images (tra, cor, sag)...')
    img_tra, img_cor, img_sag = readOriginalImages(inFolder, current_folder)
    
    # Step 2: Create output folder for ROI files (temporary processing folder)
    case_output_folder = join(outputDirectory, current_folder)
    if not os.path.exists(case_output_folder):
        os.makedirs(case_output_folder)
    
    # Step 3: Call getCroppedIsotropicImgs to process images and get ROI versions
    # Note: getCroppedIsotropicImgs expects output_folder to be a path it can append filenames to
    # It does string concatenation, so we need to ensure it ends with a separator
    print('Processing images with getCroppedIsotropicImgs...')
    output_folder_path = case_output_folder + os.sep
    roi_tra, roi_cor, roi_sag, startROI, sizeROI = utils.getCroppedIsotropicImgs(
        output_folder_path, img_tra, img_cor, img_sag
    )
    
    # Step 4: Make prediction using the CNN model
    print(f'Predicting segmentation for {current_folder}...')
    output_NN = makePrediction(model, img_size, roi_tra, roi_sag, roi_cor)
    
    # Step 5: Post-process the prediction
    print('Post-processing prediction...')
    pred_nn = sitk.GetImageFromArray(output_NN[0, :, :, :, 0])
    pred_nn = utils.binaryThresholdImage(pred_nn, threshold)
    pred_nn = utils.getLargestConnectedComponents(pred_nn)
    
    # Step 6: Save ROI segmentation
    if save_segmentations:
        print('Saving ROI segmentation...')
        sitk.WriteImage(pred_nn, join(outputDirectory, current_folder, 'predicted_roi.nrrd'))
    
    # Step 7: Upscale to original resolution if requested
    if all_params.get('orig_resolution', True):
        print('Upscaling to original resolution...')
        
        # Read the original transversal image for reference
        img_tra_original = sitk.ReadImage(join(inFolder, current_folder, 'img_tra.nrrd'))
        
        # Create HR version for upscaling (resample to 0.5mm isotropic)
        img_tra_HR = utils.resampleImage(img_tra_original, [0.5, 0.5, 0.5], 
                                        sitk.sitkLinear, 0)
        img_tra_HR = utils.sizeCorrectionImage(img_tra_HR, factor=6, imgSize=168)
        
        # Place prediction back into HR space
        output_predicted_original = sitk.Image(img_tra_HR.GetSize(), sitk.sitkFloat32)
        arr = sitk.GetArrayFromImage(output_predicted_original)
        arr[:] = 0
        
        # Map ROI prediction back to HR space
        arr[startROI[2]:startROI[2]+sizeROI[2], 
            startROI[1]:startROI[1]+sizeROI[1],
            startROI[0]:startROI[0]+sizeROI[0]] = output_NN[0, :, :, :, 0]
        
        output_predicted = sitk.GetImageFromArray(arr)
        output_predicted = utils.binaryThresholdImage(output_predicted, threshold)
        output_predicted = utils.getLargestConnectedComponents(output_predicted)
        output_predicted = sitk.BinaryFillhole(output_predicted, fullyConnected=True)
        output_predicted.SetOrigin(img_tra_HR.GetOrigin())
        output_predicted.SetDirection(img_tra_HR.GetDirection())
        output_predicted.SetSpacing(img_tra_HR.GetSpacing())
        
        if save_segmentations:
            sitk.WriteImage(output_predicted, 
                          join(outputDirectory, current_folder, 'predicted_HR.nrrd'))
        
        # Resample to original resolution
        segm_original = utils.resampleToReference(output_predicted, img_tra_original, 
                                                 sitk.sitkNearestNeighbor, 0)
        thresholded = utils.binaryThresholdImage(segm_original, threshold)
        
        # Apply post-processing for prostate segmentation
        if all_params.get('type_segmentation', 'Prostate') == 'Prostate':
            np_pred_orig = sitk.GetArrayFromImage(thresholded)
            np_pred_orig = utils.getLargestConnectedComponentsBySliceAndFillHoles(np_pred_orig)
            thresholded = sitk.GetImageFromArray(np_pred_orig)
        
        if save_segmentations:
            sitk.WriteImage(thresholded, 
                          join(outputDirectory, current_folder, 'predicted_original_resolution.nrrd'))
        
        # Create visualization
        print('Creating visualization...')
        slices = all_params.get('orig_slices', 'middle')
        title = f'{current_folder} - Segmentation Complete'
        
        utilsviz.drawSeriesItk(img_tra_original, slices=slices, title=title, 
                              contours=[thresholded], labels=['Prediction'],
                              savefig=join(outputImages, current_folder))
    
    print(f'Case {current_folder} completed successfully!')


def getProperFolders(inFolder: str, cases: Any) -> List[str]:
    """
    Gets the list of case folders to process.
    
    Args:
        inFolder: Input folder containing case directories
        cases: Either 'all' to process all cases, or a list of case numbers
        
    Returns:
        Sorted list of case folder names
    """
    if isinstance(cases, str):
        if cases == 'all':
            examples = [f for f in os.listdir(inFolder) 
                       if os.path.isdir(join(inFolder, f)) and f.startswith('Case-')]
    else:
        examples = [f'Case-{case:04d}' for case in cases]
    examples.sort()
    return examples


def makeNewSegmentation(inFolder: str, outputDirectory: str, outputImages: str,
                        model_weights_file: str, all_params: Dict[str, Any], 
                        cases: Any = 'all', save_segmentations: bool = True) -> None:
    """
    Main function to perform segmentation on new cases without ground truth.
    
    Args:
        inFolder: Input folder where all the cases are
        outputDirectory: Where to save the predictions as nrrd files
        outputImages: Where to save visualization images
        model_weights_file: Path to the model weights file (.hdf5)
        all_params: Configuration parameters dictionary
        cases: Either 'all' to process all cases, or a list of case numbers
        save_segmentations: Whether to save the segmentation files
    """
    # Read parameters
    model_type = all_params['model_name']
    img_size = 168
    threshold = 0.5
    
    # Load the model
    print('Loading model...')
    model = models.readProperModel(model_type, img_size)
    
    # Load model weights
    print(f'Loading weights from {model_weights_file}...')
    if not os.path.exists(model_weights_file):
        raise FileNotFoundError(f"Model weights file not found: {model_weights_file}")
    model.load_weights(model_weights_file)
    
    # Validate input folder
    if not os.path.exists(inFolder):
        raise FileNotFoundError(f"Input folder not found: {inFolder}")
    
    # Get list of cases to process
    examples = getProperFolders(inFolder, cases)
    if not examples:
        raise ValueError(f"No valid case folders found in {inFolder}. "
                        f"Expected folders named 'Case-XXXX' (e.g., 'Case-0001')")
    print(f'Found {len(examples)} case(s) to process: {examples}')
    
    # Create output directories
    if not os.path.exists(outputImages):
        os.makedirs(outputImages)
    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)
    
    # Process each case
    for id_folder, current_folder in enumerate(examples):
        t0 = time.time()
        try:
            processNewCase(inFolder, outputDirectory, outputImages, all_params,
                          current_folder, img_size, model, threshold, save_segmentations)
        except Exception as e:
            print(f"---------------------------- Failed {current_folder} error: {e} ----------------")
            import traceback
            traceback.print_exc()
        print(f'*** Elapsed time {time.time()-t0:0.2f} seg')


if __name__ == '__main__':
    # ============================================================================
    # MANUAL PARAMETER CONFIGURATION
    # ============================================================================
    
    # Root folder of the project
    root_folder = '/home/olmozavala/Dropbox/MyProjects/UM/Paper_ProstateAndPZ_testcode/'
    
    # Input folder: Where the case folders are located (each containing img_tra.nrrd, img_cor.nrrd, img_sag.nrrd)
    inputDirectory = join(root_folder, 'data','Test')
    
    # Output folder: Where to save the segmentation predictions
    outputDirectory = join(root_folder, 'output', 'Test', 'Prostate')
    
    # Output images folder: Where to save visualization images
    outputImages = join(root_folder, 'output', 'Test', 'Prostate')
    
    # Model weights file: Path to the trained model weights (.hdf5 file)
    model_weights_file = join(root_folder, 'models', 'Prostate', 'Siemens.hdf5')
    
    # Model configuration
    model_name = '3dm'  # Model type: '3dm' (3D Multi-stream) or '3ddropout'
    type_segmentation = 'Prostate'  # Type of segmentation: 'Prostate' or 'PZ'
    
    # Processing options
    cases = 'all'  # Which cases to process: 'all' or a list like [1, 2, 3] for Case-0001, Case-0002, etc.
    save_segmentations = True  # Whether to save the segmentation .nrrd files
    orig_resolution = True  # If True, also compute predictions at original resolution (not just ROI)
    
    # Visualization options
    display_images = True # If True, displays images on the fly (requires display)
    orig_slices = 'middle'  # Which slices to visualize: 'all', 'middle', or array like [80, 90, 100]
    
    # ============================================================================
    # BUILD PARAMETER DICTIONARY
    # ============================================================================
    all_params = {
        'model_name': model_name,
        'type_segmentation': type_segmentation,
        'display_images': display_images,
        'orig_resolution': orig_resolution,
        'orig_slices': orig_slices,
        'cases': cases,
        'save_segmentations': save_segmentations
    }
    
    # Set visualization flag
    utilsviz.view_results = display_images
    
    # ============================================================================
    # RUN SEGMENTATION
    # ============================================================================
    print("=" * 70)
    print("NEW CASE SEGMENTATION")
    print("=" * 70)
    print(f"Input directory: {inputDirectory}")
    print(f"Output directory: {outputDirectory}")
    print(f"Model weights: {model_weights_file}")
    print(f"Model type: {model_name}")
    print(f"Segmentation type: {type_segmentation}")
    print(f"Cases to process: {cases}")
    print("=" * 70)
    
    makeNewSegmentation(inputDirectory, outputDirectory, outputImages, 
                      model_weights_file, all_params, cases=cases, 
                      save_segmentations=save_segmentations)

