from os.path import join

_test_folder='Siemens' # It can be GE or Siemens
_test_model='Siemens' # It can be 'GE','Siemens','Siemens_and_GE'
_type_seg = 'Prostate' # It can be 'Prostate' or 'PZ'
_root_folder = '../'
_data_folder= join(_root_folder, _test_folder)
#_data_folder= F'/data/UM/ProstateSegdata/{_test_folder}'
_output_folder = F'{_root_folder}/output/{_test_folder}'

def getMakeSegmentationConfig():
    cur_config_multiple = {
        'input_folder': _data_folder, # Where the cases are. Enumerated and in .nrrd format
        'output_folder': join(_output_folder,_type_seg,_test_model,'segmentations'), # Where to save segmentations
        'output_images': join(_output_folder,_type_seg,_test_model,'images'),
        'weights': join(_root_folder,'models',_type_seg,F'{_test_model}.hdf5'),
        'model_name': '3dm',
        'type_segmentation': _type_seg,  # ('Prostate' or 'PZ')
        'display_images': False, # Displays generated images on the fly
        # 'roi_slices': np.arange(80,96,4),  # Which slices to generate for ROI ('all', 'middle', or an array of numbers)
        'roi_slices': 'middle',  # Which slices to generate for ROI ('all', 'middle', or an array of numbers)
        'save_segmentations': True, # Indicates if the segmentations files are going to be saved (normally yes)
        'orig_resolution': True, # If true, the images and segmentation will be computed at the original resolution (not only ROI)
        'orig_slices': 'middle',# Which slices to generate ('all', 'middle', or an array of numbers) Only for original
        'cases':'all'
        }

    return cur_config_multiple
