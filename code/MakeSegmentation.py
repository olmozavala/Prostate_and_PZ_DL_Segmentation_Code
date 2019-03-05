import preproc.utils as utils
import visualization.utilsviz as utilsviz
import models.models as models
from models.metrics import numpy_dice
import os
from os.path import join, getmtime
import SimpleITK as sitk
from pandas import DataFrame
from config.MainConfig import getMakeSegmentationConfig
from inout.io_mri import readImgAndProstate, readPZ, readROI
import time

def getLatestFile(file_names):
    latest_file = ''
    largest_date = -1
    for cur_file in file_names:
        cur_time = getmtime(cur_file)
        if cur_time > largest_date:
            largest_date = cur_time
            latest_file = cur_file

    return latest_file

def makePrediction(model, roi_ctr_pro, img_size, roi_img1, roi_img2, roi_img3):
        input_array = utils.createInputArray(True, img_size, roi_img1, roi_img2, roi_img3)
        output_NN = model.predict(input_array, verbose=1)

        return output_NN

def saveMetricsData(dsc_data, m_names, outputImages, save_file=True):
    # ************** Plot and save Metrics for ROI *****************
    dsc_data.loc['AVG'] = dsc_data.mean() # Compute all average values
    title = F"DSC AVG ROI: {dsc_data.loc['AVG'][m_names.get('dsc_roi')]:.3f}, " \
            F"AVG Original {dsc_data.loc['AVG'][m_names.get('dsc_orig')]:.3f}"
    utilsviz.plotMultipleBarPlots([dsc_data[m_names.get('dsc_roi')].dropna().to_dict(),
                                   dsc_data[m_names.get('dsc_orig')].dropna().to_dict() ],
                                  title=title,
                                  legends=['ROI','Original'], savefig=join(outputImages,'aBoth_DSC.png'))
    dsc_data.to_csv(join(outputImages,'all_DSC.csv'))

def procSingleCase(inFolder, outputDirectory, outputImages, all_params, current_folder, multistream,
                   img_size, model, threshold, roi_slices, dsc_data, m_names, save_segmentations=True, indx=0):
    '''
    This function process the segmentation of a single case. It makes the NN predition and saves the results
    :param inFolder:  Input folder where all the cases are
    :param outputDirectory: Where to save the prediction as nrrd file
    :param outputImages: Where to save the images and CSV files
    :param all_params: Parameters from the user (currently used only slices)
    :param current_folder: Current case
    :param multistream: Using multistream model
    :param img_size: Image size
    :param model: TensorFlow model
    :param threshold: Threshold to binarize the images
    :param roi_slices:
    :param dsc_data: DataFrame with the current metrics for all the cases
    :param m_names: String array with the names of the metrics like DSC, etc.
    :param save_segmentations: Boolean: indicate if we want to save the results of the NN
    :param indx: Integer of the current number of the folder, used to save the results only every 10 cases
    :return:
    '''
    # Reads original image and prostate
    [img_tra_original, img_tra_HR, ctr_pro, ctr_pro_HR, roi_ctr_pro, startROI, sizeROI] = readImgAndProstate(inFolder, current_folder)
    np_ctr_pro = sitk.GetArrayViewFromImage(ctr_pro)
    np_roi_ctr_pro = sitk.GetArrayViewFromImage(roi_ctr_pro)
    # Reads PZ and input for NN
    if type_segmentation == 'PZ':
        [ctr_pz, ctr_pz_HR, roi_ctr_pz] = readPZ(inFolder, current_folder, multistream, img_size)
        np_ctr_pz = sitk.GetArrayViewFromImage(ctr_pz)
        np_roi_ctr_pz = sitk.GetArrayViewFromImage(roi_ctr_pz)

    [roi_img1, roi_img2, roi_img3] = readROI(inFolder, current_folder, type_segmentation)

    print('Predicting image {} ({})....'.format(current_folder, inFolder))
    output_NN = makePrediction(model, roi_ctr_pro, img_size, roi_img1, roi_img2, roi_img3)

    # ************** Binary threshold and largest connected component ******************
    print('Threshold and largest component...')
    pred_nn = sitk.GetImageFromArray(output_NN[0,:,:,:,0])
    pred_nn = utils.binaryThresholdImage(pred_nn, threshold)
    pred_nn = utils.getLargestConnectedComponents(pred_nn)
    np_pred_nn = sitk.GetArrayViewFromImage(pred_nn)

    # ************** Compute metrics for ROI ******************
    c_img_folder = join(outputImages,current_folder)
    print('Metrics...')
    if type_segmentation == 'Prostate':
        cur_dsc_roi = numpy_dice(np_roi_ctr_pro, np_pred_nn)
    if type_segmentation == 'PZ':
        cur_dsc_roi = numpy_dice(np_roi_ctr_pz, np_pred_nn)
    print(F'--------------{c_img_folder} DSC ROI: {cur_dsc_roi:02.2f}  ------------')

    # ************** Visualize and save results for ROI ******************
    slices = roi_slices
    title = F'DSC {cur_dsc_roi:02.3f}'
    print('Making ROI images...')
    if type_segmentation == 'Prostate':
        utilsviz.drawMultipleSeriesItk([roi_img1], slices=slices, subtitles=[title], contours=[roi_ctr_pro, pred_nn],
                            savefig=join(outputImages,'ROI_PROSTATE_'+current_folder), labels=['GT','NN'])
    if type_segmentation == 'PZ':
        utilsviz.drawMultipleSeriesItk([roi_img1], slices=slices, subtitles=[title], contours=[roi_ctr_pro, roi_ctr_pz, pred_nn],
                             savefig=join(outputImages,'ROI_PZ_'+current_folder), labels=['Prostate','GT','NN'])

    # ************** Save ROI segmentation *****************
    if save_segmentations:
        print('Saving original prediction (ROI)...')
        if not os.path.exists(join(outputDirectory, current_folder)):
            os.makedirs(join(outputDirectory, current_folder))
        sitk.WriteImage(pred_nn, join(outputDirectory, current_folder, 'predicted_roi.nrrd'))

    dsc_data.loc[current_folder][m_names.get('dsc_roi')] = cur_dsc_roi

    # ************** Plot and save Metrics for ROI *****************
    print('Making Bar plots ...')
    dsc_data.loc['AVG'] = dsc_data.mean() # Compute all average values
    title = F"DSC AVG ROI: {dsc_data.loc['AVG'][m_names.get('dsc_roi')]:.3f}"
    utilsviz.plotMultipleBarPlots([dsc_data[m_names.get('dsc_roi')].dropna().to_dict()], title=title,
                                  legends=['ROI'], savefig=join(outputImages,'aroi_DSC.png'))
    dsc_data.to_csv(join(outputImages,'aroi_DSC.csv'))

    # ************** Compute everything but for the original resolution *****************
    if not(only_ROI): # in this case we do not upscale, just show the prediction
        print('Getting Original resolution ...')
        output_predicted_original = sitk.Image(img_tra_HR.GetSize(), sitk.sitkFloat32)
        arr = sitk.GetArrayFromImage(output_predicted_original) # Gets an array same size as original image
        arr[:] = 0 # Make everything = 0
        arr[startROI[2]:startROI[2]+sizeROI[2], startROI[1]:startROI[1]+sizeROI[1],startROI[0]:startROI[0]+sizeROI[0]] = output_NN[0,:,:,:,0]
        output_predicted = sitk.GetImageFromArray(arr)
        output_predicted = utils.binaryThresholdImage(output_predicted, threshold)
        output_predicted = utils.getLargestConnectedComponents(output_predicted)
        output_predicted = sitk.BinaryFillhole(output_predicted, fullyConnected=True)
        output_predicted.SetOrigin(img_tra_HR.GetOrigin())
        output_predicted.SetDirection(img_tra_HR.GetDirection())
        output_predicted.SetSpacing(img_tra_HR.GetSpacing())

        if save_segmentations:
            sitk.WriteImage(output_predicted, join(outputDirectory, current_folder, 'predicted_HR.nrrd'))

        segm_dis = utils.resampleToReference(output_predicted, img_tra_original, sitk.sitkNearestNeighbor, 0)
        thresholded = utils.binaryThresholdImage(segm_dis, threshold)
        np_pred_nn_orig = sitk.GetArrayFromImage(thresholded)

        if type_segmentation == 'Prostate':
            np_pred_nn_orig = utils.getLargestConnectedComponentsBySliceAndFillHoles(np_pred_nn_orig)

        thresholded = sitk.GetImageFromArray(np_pred_nn_orig)

        if save_segmentations:
            sitk.WriteImage(thresholded, join(outputDirectory, current_folder, 'predicted_original_resolution.nrrd'))

        if type_segmentation == 'Prostate':
            cur_dsc_original = numpy_dice(np_ctr_pro, np_pred_nn_orig)
        if type_segmentation == 'PZ':
            cur_dsc_original = numpy_dice(np_ctr_pz, np_pred_nn_orig)

        title = '{} DSC {:02.3f}'.format(current_folder, cur_dsc_original)
        slices = all_params['orig_slices']
        print('Making Original images ...')
        if type_segmentation == 'PZ':
            utilsviz.drawSeriesItk(img_tra_original, slices=slices, title=title, contours=[ctr_pro, ctr_pz, thresholded],
                                   labels=['Prostate','PZ','NN'], savefig=join(outputImages, current_folder))
        if type_segmentation == 'Prostate':
            utilsviz.drawSeriesItk(img_tra_original, slices=slices, title=title, contours=[ctr_pro, thresholded],
                                   labels=['GT','NN'], savefig=join(outputImages, current_folder))

    dsc_data.loc[current_folder][m_names.get('dsc_orig')] = cur_dsc_original

    return dsc_data

def getProperFolders(inFolder, cases):
    '''Depending on the value of cases it reads the proper folders from the list of folders'''
    # *********** Define which cases are we going to perform the segmentation **********
    if isinstance(cases,str):
        if cases == 'all':
            examples = os.listdir(inFolder)
    else:
        examples = ['Case-{:04d}'.format(case) for case in cases]
    examples.sort()

    return examples

def makeSegmentation(inFolder, outputDirectory, outputImages, model_weights_file,
                        all_params, cases='all', save_segmentations=True):
    '''
    This function computes a new mask from the spedified model weights and model
    :param inFolder:
    :param outputDirectory:
    :param outputImages:
    :param model_weights_file:
    :param all_params:
    :param cases:
    :param save_segmentations:
    :return:
    '''
    # *********** Reads the parameters ***********
    model_type = all_params['model_name']
    roi_slices = all_params['roi_slices']

    img_size = 168
    threshold = 0.5

    # *********** Chooses the proper model ***********
    print('Reading model ....')
    model = models.readProperModel(model_type, img_size)
    multistream = True

    # *********** Reads the weights***********
    print('Reading weights ....')
    model.load_weights(model_weights_file)

    examples = getProperFolders(inFolder, cases)

    if not os.path.exists(outputImages):
        os.makedirs(outputImages)

    # *********** Makes a dataframe to contain the DSC information **********
    m_names = {'dsc_roi':'ROI','dsc_orig':'Original'}

    # Check if the output fiels already exist, in thtat case read the df from it.
    dsc_data = DataFrame(index = examples, columns=[m_names.get('dsc_roi'),m_names.get('dsc_orig')])

    # *********** Iterates over each case *********
    for id_folder, current_folder in enumerate(examples):
        t0 = time.time()
        try:
            dsc_data = procSingleCase(inFolder, outputDirectory, outputImages, all_params, current_folder, multistream,
                   img_size, model, threshold, roi_slices, dsc_data, m_names, save_segmentations, indx=id_folder)

            saveMetricsData(dsc_data, m_names, outputImages)
        except Exception as e:
            print("---------------------------- Failed {} error: {} ----------------".format(current_folder, e))
        print(F'*** Elapsed time {time.time()-t0:0.2f} seg')

if __name__ == '__main__':

    all_params = getMakeSegmentationConfig()

    only_ROI = not(all_params['orig_resolution'])
    type_segmentation =  all_params['type_segmentation']
    model_name = all_params['model_name']
    disp_images = all_params['display_images']
    cases = all_params['cases']
    save_segmentations = all_params['save_segmentations']
    utilsviz.view_results = disp_images

    outputImages = all_params['output_images']
    inputDirectory = all_params['input_folder']
    outputDirectory = all_params['output_folder']
    model_weights_file = all_params['weights']
    makeSegmentation(inputDirectory, outputDirectory, outputImages, model_weights_file,
                     all_params, cases=cases, save_segmentations=save_segmentations)

