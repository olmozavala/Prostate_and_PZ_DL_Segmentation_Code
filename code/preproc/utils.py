import SimpleITK as sitk
import visualization.utilsviz as utilsviz
import math
import os
from os.path import join
import numpy as np
import cv2
from skimage import measure
from scipy.ndimage.morphology import binary_fill_holes


def resampleImage(in_img, new_spacing, interpolator, defaultValue):
    ''' Resamples the image to new spacing
    :param in_img:
    :param new_spacing:
    :param interpolator:
    :param defaultValue:
    :return:
    '''

    # Cast image to float
    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
    in_img = castImageFilter.Execute(in_img)

    original_spacing = in_img.GetSpacing()
    original_size = in_img.GetSize()
    new_size = [int(round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
                int(round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
                int(round(original_size[2] * (original_spacing[2] / new_spacing[2])))]

    ## Compute the resampling
    filter = sitk.ResampleImageFilter()

    # "in_img, out size, tranform?, interpolation, out origin, out space, out direction, pixel type id, "
    out_img = filter.Execute(in_img, new_size, sitk.Transform(), interpolator, in_img.GetOrigin(),
                                  new_spacing, in_img.GetDirection(), defaultValue, in_img.GetPixelIDValue())

    return out_img

def resampleToReference(inputImg, referenceImg, interpolator, defaultValue):
    '''
    Resamples an image to a reference image
    :param inputImg:
    :param referenceImg:
    :param interpolator:
    :param defaultValue:
    :return:
    '''

    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
    inputImg = castImageFilter.Execute(inputImg)

    filter = sitk.ResampleImageFilter()
    filter.SetReferenceImage(referenceImg)
    filter.SetDefaultPixelValue(float(defaultValue))

    filter.SetInterpolator(interpolator)
    outImage = filter.Execute(inputImg)

    return outImage

def resampleData(orig_imgs, orig_ctrs, resampling, ctr_names_tmp = ['NONE'], optical_flow_interpolation = True):
    '''
    Resamples images and contours to specific resolution.
    :param orig_imgs: itk images to resample
    :param orig_ctrs: itk controus to resmaple
    :param resampling: array of floats indicating the resampling resolution in mm
    :param ctr_names_tmp: array of names of the ctrs, used only for printing errors
    :param optical_flow_interpolation: bool indicating if we want to do optical flow on the interpolation
    :return:
    '''

    tot_ctrs = len(orig_ctrs)
    tot_imgs = len(orig_imgs)
    interpolator = sitk.sitkLinear

    # These 2 vars hold the high resolution images and contours
    hr_imgs = []
    hr_ctrs = []

    # This is the resampling of the FIRST Image (tra)
    hr_imgs.append(resampleImage(orig_imgs[0], resampling, interpolator, 0))
    hr_imgs[0] = sizeCorrectionImage(hr_imgs[0], 6, 168) # Makes the image to size 168 and multiple of 6

    # Resampling all other images with respect to the first image
    for img_idx in range(1,tot_imgs):
        hr_imgs.append(resampleToReference(orig_imgs[img_idx], hr_imgs[0], interpolator, 0))

    # *************** Interpolation between images using OptialFlow (Nearest Neighbor if it fails)
    if optical_flow_interpolation:
        print('\t Optical flow ....')
        interpolator = sitk.sitkNearestNeighbor # Default interpolater (only if optical flow fails)
        # Iterate over all the contours
        for ctr_idx in range(tot_ctrs):
            # Resample to reference using nearest neighbor
            tempbk = sitk.GetArrayFromImage(resampleToReference(orig_ctrs[ctr_idx], hr_imgs[0], interpolator, 0))
            temp = tempbk.copy()

            if optical_flow_interpolation: # Check if we will do the interpolation
                totslices = temp.shape[0]
                try:
                    idSlice = 0
                    while np.sum(temp[idSlice,:,:],axis=(0,1)) == 0: # Initialize first 2 layers to work with
                        idSlice+=1
                    prevLay = idSlice
                    nextLay = idSlice+1

                    while (nextLay < totslices) and (np.sum(temp[nextLay,:,:], axis=(0,1)) != 0): # Verify the 'next layer' is not empty (done)
                        # utilsviz.drawMultipleSeriesNumpy([temp[prevLay:nextLay+1,:,:]], slices='all', title='', contours=[], savefig='', labels=[])
                        if np.sum(temp[prevLay,:,:] - temp[nextLay,:,:],axis=(0,1)) != 0: # Check there is difference between layers
                            dist_intra_slice = nextLay-prevLay
                            startLay = prevLay+int(np.floor(dist_intra_slice/2))-1
                            endLay = nextLay+int(np.floor(dist_intra_slice/2))-1
                            # print('P: {}, N:{}'.format(prevLay, nextLay))
                            # print('S: {}, E:{}'.format(startLay, endLay))
                            # Do our stuff between prev and next
                            prev = temp[startLay,:,:].astype(np.uint8)*255
                            next = temp[endLay,:,:].astype(np.uint8)*255
                            img, ctrs1, _ = cv2.findContours(prev, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                            if len(np.shape(ctrs1)) == 4:
                                ctrs1 = np.array(ctrs1)[0,:,0,:]
                                # Obtain optical flow
                                flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                                orig = ctrs1.copy() # Copy original position of contours
                                for idSubSlice in np.arange(1,dist_intra_slice): # Move with the flow
                                    # print('Ctr_idx {}  {}-{}'.format(ctr_idx, prevLay, nextLay))
                                    for idCtr in range(len(ctrs1)):
                                        ctrs1[idCtr:] = orig[idCtr,:] + (idSubSlice/dist_intra_slice)*flow[orig[idCtr,1], orig[idCtr,0]].T
                                    newImg = np.zeros(next.shape)
                                    cv2.fillPoly(newImg, pts = [ctrs1], color=(255,255,255))
                                    temp[startLay+idSubSlice,:,:] = newImg

                            prevLay = endLay
                            nextLay = endLay+1

                        nextLay +=1
                except Exception as e:
                    print("!!!!!! Failed Interpolation for ctr {} Using nearest neighbor instead. Error: {} !!!!!!".format(ctr_names_tmp[ctr_idx], e))
                    temp = tempbk # Restore original resampled version with nearest neighbor

            hr_ctrs.append(sitk.GetImageFromArray(temp))
            hr_ctrs[ctr_idx] = binaryThresholdImage(hr_ctrs[ctr_idx], 0.00001)

    return [hr_imgs, hr_ctrs]

def saveImage(img, out_folder, img_name):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    sitk.WriteImage(img, join(out_folder, img_name))

def saveImages(output_folder, pretxt,imgs, img_names):
    '''
    :param output_folder: str path to save the image
    :param pretxt: str prefix to store for each img name
    :param imgs: itk array of images
    :param img_names: str array of image names
    :return:
    '''
    print("Saving images ({})...".format(pretxt))
    for img_idx in range(len(imgs)):
        file_name = '{}_{}.nrrd'.format(pretxt, img_names[img_idx])
        saveImage(imgs[img_idx], output_folder, file_name)

def sizeCorrectionImage(img, factor, imgSize):
    # corrects the size of an image to a multiple of the factor
    # assumes that input image size is larger than minImgSize, except for z-dimension
    # factor is important in order to resample image by 1/factor (e.g. due to slice thickness) without any errors

    # print("Resizing to {}x{}x{}".format(imgSize, imgSize, imgSize))
    size = img.GetSize()
    correction = False
    # check if bounding box size is multiple of 'factor' and correct if necessary
    # x-direction
    if (size[0])%factor != 0:
        cX = factor-(size[0]%factor)
        correction = True
    else:
        cX = 0
    # y-direction
    if (size[1])%factor != 0:
        cY = factor-(size[1]%factor)
        correction = True
    else:
        cY  = 0

    if (size[2]) !=imgSize:
        cZ = (imgSize-size[2])
        # if z image size is larger than maxImgsSize, crop it (customized to the data at hand. Better if ROI extraction crops image)
        if cZ <0:
            # print('image gets filtered')
            cropFilter = sitk.CropImageFilter()
            cropFilter.SetUpperBoundaryCropSize([0,0,int(math.floor(-cZ/2))])
            cropFilter.SetLowerBoundaryCropSize([0,0,int(math.ceil(-cZ/2))])
            img = cropFilter.Execute(img)
            cZ=0
        else:
            correction = True
    else:
        cZ = 0

    # if correction is necessary, increase size of image with padding (default value for padding = -1)
    if correction:
        filter = sitk.ConstantPadImageFilter()
        filter.SetPadLowerBound([int(math.floor(cX/2)), int(math.floor(cY/2)), int(math.floor(cZ/2))])
        filter.SetPadUpperBound([int(math.ceil(cX/2)), int(math.ceil(cY)), int(math.ceil(cZ/2))])
        # filter.SetConstant(-1)
        filter.SetConstant(0)  # Here Anneke was making a pad with -1 (change it to 0)
        # EL PROBLEMA ES QUE CUANDO SE APLICA ESTE FILTRO SE PONE BLANCA Y ESO NO PASA CON PROSTATEX
        utilsviz.drawSeriesItk(img, slices='middle', title='SOP Befor resampling')
        outPadding = filter.Execute(img)
        utilsviz.drawSeriesItk(outPadding, slices='middle', title='SOP AFTER resampling')
        return outPadding
    else:
        return img

def normalizeIntensitiesPercentile(imgs):
    out = []

    normalizationFilter = sitk.IntensityWindowingImageFilter()

    # This part normalizes the images
    for idx in range(len(imgs)):
        img = imgs[idx]
        array = np.ndarray.flatten(sitk.GetArrayFromImage(img))

        # Gets the value of the specified percentiles
        upperPerc = np.percentile(array, 99) #98
        lowerPerc = np.percentile(array, 1) #2

        normalizationFilter.SetOutputMaximum(1.0)
        normalizationFilter.SetOutputMinimum(0.0)
        normalizationFilter.SetWindowMaximum(upperPerc)
        normalizationFilter.SetWindowMinimum(lowerPerc)

        floatImg= sitk.Cast(img, sitk.sitkFloat32) # Cast to float

        # ALL images get normalized between 0 and 1
        outNormalization = normalizationFilter.Execute(floatImg) #Normalize to 0-1
        out.append(outNormalization)

        # If you want to see the differences before and after normalization
        # utilsviz.drawSeriesItk(floatImg, slices=[90], title='', contours=[], savefig='', labels=[])
        # utilsviz.drawSeriesItk(out[-1], slices=[90], title='', contours=[], savefig='', labels=[])
    return out

def binaryThresholdImage(img, lowerThreshold):
    '''
    Obtains a binary image.
    :param img: Itk image
    :param lowerThreshold: Lower threshold to use as valid value.
    :return:
    '''

    maxFilter = sitk.StatisticsImageFilter()
    maxFilter.Execute(img)
    maxValue = maxFilter.GetMaximum()
    thresholded = sitk.BinaryThreshold(img, lowerThreshold, maxValue, 1, 0)

    return thresholded

def castImage(img, type):

    castFilter = sitk.CastImageFilter()
    castFilter.SetOutputPixelType(type) #sitk.sitkUInt8
    out = castFilter.Execute(img)

    return out

def sizeCorrectionBoundingBox(bbox, newSize, factor):
    # adapt the start index of the ROI to the manual bounding box size
    # (assumes that all ROIs are smaller than newSize pixels in length and width)
    # correct the start index according to the new size of the bounding box
    start = bbox[0:3]
    start = list(start)
    size = bbox[3:6]
    size = list(size)
    start[0] = start[0] - math.floor((newSize - size[0]) / 2)
    start[1] = start[1] - math.floor((newSize - size[1]) / 2)

    # check if BB start can be divided by the factor (essential if ROI needs to be extracted from non-isotropic image)
    if (start[0]) % factor != 0:
        cX = (start[0] % factor)
        newStart = start[0] - cX
        start[0] = int(newStart)

    # y-direction
    if (start[1]) % factor != 0:
        cY = (start[1] % factor)
        start[1] = int(start[1] - cY)

    size[0] = newSize
    size[1] = newSize

    return start, size

def getBoundingBox(img):
    '''
    Gets the bbox with values from an image
    :param img:
    :return:
    '''

    masked = binaryThresholdImage(img, 0.1)
    statistics = sitk.LabelShapeStatisticsImageFilter()
    statistics.Execute(masked)

    bb = statistics.GetBoundingBox(1)

    return bb

def getMaximumValue(img):

    maxFilter = sitk.StatisticsImageFilter()
    maxFilter.Execute(img)
    maxValue = maxFilter.GetMaximum()
    return maxValue

def thresholdImage(img, lowerValue, upperValue, outsideValue):

    thresholdFilter = sitk.ThresholdImageFilter()
    thresholdFilter.SetUpper(upperValue)
    thresholdFilter.SetLower(lowerValue)
    thresholdFilter.SetOutsideValue(outsideValue)

    out = thresholdFilter.Execute(img)
    return out

def getCroppedIsotropicImgs(output_folder, *imgs):

    img_tra = imgs[0]
    img_cor = imgs[1]
    img_sag = imgs[2]

    # normalize intensities
    print('... normalize intensities ...')
    img_tra, img_cor, img_sag = normalizeIntensitiesPercentile([img_tra, img_cor, img_sag])

    # get intersecting region (bounding box)
    print('... get intersecting region (ROI) ...')

    # upsample transversal image to isotropic voxel size (isotropic transversal image coordinate system is used as reference coordinate system)
    tra_HR = resampleImage(img_tra, [0.5, 0.5, 0.5], sitk.sitkLinear,0)
    tra_HR = sizeCorrectionImage(tra_HR, factor=6, imgSize=168)

    # resample coronal and sagittal to tra_HR space
    # resample coronal to tra_HR and obtain mask (voxels that are defined in coronal image )
    cor_toTraHR = resampleToReference(img_cor, tra_HR, sitk.sitkLinear,-1)
    cor_mask = binaryThresholdImage(cor_toTraHR, 0)

    tra_HR_Float = castImage(tra_HR, sitk.sitkFloat32)
    cor_mask_Float = castImage(cor_mask, sitk.sitkFloat32)
    # mask transversal volume (set voxels, that are defined only in transversal image but not in coronal image, to 0)
    coronal_masked_traHR = sitk.Multiply(tra_HR_Float, cor_mask_Float)

    # resample sagittal to tra_HR and obtain mask (voxels that are defined in sagittal image )
    sag_toTraHR = resampleToReference(img_sag, tra_HR, sitk.sitkLinear,-1)
    sag_mask = binaryThresholdImage(sag_toTraHR, 0)
    # mask sagittal volume
    sag_mask_Float = castImage(sag_mask, sitk.sitkFloat32)

    # masked image contains voxels, that are defined in tra, cor and sag images
    maskedImg = sitk.Multiply(sag_mask_Float, coronal_masked_traHR)
    bbox = getBoundingBox(maskedImg)

    # correct the size and start position of the bounding box according to new size
    start, size = sizeCorrectionBoundingBox(bbox, newSize=168, factor=6)
    start[2] = 0
    size[2] = tra_HR.GetSize()[2]

    # resample cor and sag to isotropic transversal image space
    cor_traHR = resampleToReference(img_cor, tra_HR, sitk.sitkLinear, -1)
    sag_traHR = resampleToReference(img_sag, tra_HR, sitk.sitkLinear,-1)

    ## extract bounding box for all planes
    region_tra = sitk.RegionOfInterest(tra_HR, [size[0], size[1], size[2]],
                                       [start[0], start[1], start[2]])
    maxVal = getMaximumValue(region_tra)
    region_tra = thresholdImage(region_tra, 0, maxVal, 0)

    region_cor = sitk.RegionOfInterest(cor_traHR, [size[0], size[1], size[2]],
                                       [start[0], start[1], start[2]])
    maxVal = getMaximumValue(region_cor)
    region_cor = thresholdImage(region_cor, 0, maxVal, 0)

    region_sag = sitk.RegionOfInterest(sag_traHR, [size[0], size[1], size[2]],
                                       [start[0], start[1], start[2]])
    maxVal = getMaximumValue(region_sag)
    region_sag = thresholdImage(region_sag, 0, maxVal, 0)

    if not os.path.exists(output_folder+ '/ROI/'):
        os.makedirs(output_folder+ '/ROI/')

    sitk.WriteImage(region_tra, output_folder + 'roi_tra.nrrd')
    sitk.WriteImage(region_cor, output_folder + 'roi_cor.nrrd')
    sitk.WriteImage(region_sag, output_folder + 'roi_sag.nrrd')
    return region_tra, region_cor, region_sag, start, size

def getCroppedIsotropicImgsOZ(imgs, ctrs, norm_img_perc, img_size=168):
    '''
    Crops the images with the intersection of Sag, Transversal and Coronal
    :param output_folder:
    :param imgs: VERY important it is assumed that the first 3 images are tra, cor, sag
    :param ctrs:
    :return:
    '''
    tot_imgs = len(imgs)
    tot_ctrs = len(ctrs)

    assert tot_imgs >= 3, 'Less than 3 images to compute the isotropic croping'
    masks = []
    # Obtain a mask for each image (ones everything above 0)
    for img_idx in range(3): # We NEED the 3 first images (AX,SAG,COR) or it will not work
        masks.append(binaryThresholdImage(imgs[img_idx], 0.0001))

    # WARNING this part assumes the order tra, cor, sag
    mask_cor_tra = sitk.Multiply(masks[0], masks[1]) # Intersection tra and cor
    mask_all = sitk.Multiply(masks[2], mask_cor_tra) # Intersection tra, cor and sag
    bbox = getBoundingBox(mask_all)

    # Obtains the start positions and size of the image to cut (size should always be 168^3
    start, size = sizeCorrectionBoundingBox(bbox, newSize=img_size , factor=6)

    roi_imgs = []
    roi_ctrs = []

    # Cuts all the images to the ROI
    # print([x.GetSize() for x in imgs])
    # utilsviz.drawMultipleSeriesItk(imgs, slices='middle', contours=mask_all)
    for img_idx in range(tot_imgs):
        roi_imgs.append( sitk.RegionOfInterest(imgs[img_idx], [size[0], size[1], size[2]], [start[0], start[1], start[2]]))
    # utilsviz.drawMultipleSeriesItk(roi_imgs, slices='middle')

    for ctr_idx in range(tot_ctrs):
        roi_ctrs.append( sitk.RegionOfInterest(ctrs[ctr_idx], [size[0], size[1], size[2]], [start[0], start[1], start[2]]))


    return roi_imgs, roi_ctrs, start, size

def getLargestConnectedComponents(img):

    connectedFilter = sitk.ConnectedComponentImageFilter()
    connectedComponents = connectedFilter.Execute(img)

    labelStatistics = sitk.LabelShapeStatisticsImageFilter()
    labelStatistics.Execute(connectedComponents)
    nrLabels = labelStatistics.GetNumberOfLabels()

    biggestLabelSize = 0
    biggestLabelIndex = 1
    for i in range(1, nrLabels+1):
        curr_size = labelStatistics.GetNumberOfPixels(i)
        if curr_size > biggestLabelSize:
            biggestLabelSize = curr_size
            biggestLabelIndex = i

    largestComponent = sitk.BinaryThreshold(connectedComponents, biggestLabelIndex, biggestLabelIndex)

    return largestComponent

def getLargestConnectedComponentsBySliceAndFillHoles(img):
    '''
    Obtains the largest connected component, but in a 2D fassion slice by slice
    :param img:
    :return:
    '''
    for ii in range(img.shape[0]):# Iterate each slice
        if np.sum(img[ii,:,:]) > 0:
            # import matplotlib.pyplot as plt
            # plt.imshow(img[ii,:,:])
            # plt.show()
            # Obtain connected component
            all_labels = measure.label(img[ii,:,:], background=0)
            if all_labels.max() > 1: # Check if there is more than one connected component
                larg_label = -1
                larg_val = -1
                for jj in range(1,all_labels.max()+1): # Select the largest component
                    cur_sum = np.sum(all_labels == jj)
                    if  cur_sum > larg_val:
                        larg_label = jj
                        larg_val = cur_sum
                # Remove everything except the largest component
                all_labels[all_labels != larg_label] = 0
                all_labels[all_labels == larg_label] = 1
                img[ii,:,:] = all_labels
            # Fill the holes
            img[ii,:,:] = binary_fill_holes(img[ii,:,:])
            # plt.imshow(img[ii,:,:])
            # plt.show()

    return img


def createInputArray( multiplane_array, img_size, *imgs):

    print('... save images to numpy array ...')
    if multiplane_array:
        outArray = [np.zeros([1, img_size, img_size, img_size, 1], dtype=np.float32),
                    np.zeros([1, img_size, img_size, img_size, 1], dtype=np.float32),
                    np.zeros([1, img_size, img_size, img_size, 1], dtype=np.float32)]
    else:
        outArray = np.zeros([1, img_size, img_size, img_size, 1], dtype=np.float32)


    # START DELETE THIS PART, ONLY TESTING IF IT WORKS FOR DIFFERENT RESOLUTIONS
    # *********************************************
    # from scipy.interpolate import RegularGridInterpolator
    # prev_size = 168
    # new_size = img_size
    # slice = 80
    #
    # nxs = np.linspace(1, new_size, new_size)
    # nys = np.linspace(1, new_size, new_size)
    # nzs = np.linspace(1, new_size, new_size)
    #
    # x = np.linspace(1, new_size, prev_size)
    # y = np.linspace(1, new_size, prev_size)
    # z = np.linspace(1, new_size, prev_size)
    #
    # allpts = np.zeros((new_size * new_size * new_size, 3))
    # ix = 0
    # for i, nx in enumerate(nxs):
    #     for j, ny in enumerate(nys):
    #         for k, nz in enumerate(nzs):
    #             allpts[ix, :] = [nx, ny, nz]
    #             ix += 1
    #
    # for ii,img in enumerate(imgs):
    #     data = sitk.GetArrayFromImage(imgs[ii])
    #     my_interp_func = RegularGridInterpolator((x, y, z), data)
    #     new_data_flat = my_interp_func(allpts)
    #     outArray[ii][0,:,:,:,0] = np.reshape(new_data_flat, (new_size, new_size, new_size))
    #
    #     # plt.imshow(new_data[slice, :, :])
    #     # plt.show()
    # END DELETE THIS PART, ONLY SEEN IF IT WORKS FOR DIFFERENT RESOLUTIONS
    # *********************************************

    if multiplane_array:
        # transversal image
        outArray[0][0, :, :, :, 0] = sitk.GetArrayFromImage(imgs[0])
        # sagittal image
        outArray[1][0, :, :, :, 0] = sitk.GetArrayFromImage(imgs[1])
        # coronal image
        outArray[2][0, :, :, :, 0] = sitk.GetArrayFromImage(imgs[2])
    else:
        outArray[0, :, :, :, 0] = sitk.GetArrayFromImage(imgs[0])

    return outArray

def createInputArrayOLD( multiplane_array, img_size, *imgs):

    print('... save images to numpy array ...')
    if multiplane_array:
        outArray = np.zeros([3, img_size, img_size, img_size, 1], dtype=np.float32)
    else:
        outArray = np.zeros([1, img_size, img_size, img_size, 1], dtype=np.float32)

    # transversal image
    outArray[0, :, :, :, 0] = sitk.GetArrayFromImage(imgs[0])
    if multiplane_array:
        # sagittal image
        outArray[1, :, :, :, 0] = sitk.GetArrayFromImage(imgs[1])
        # coronal image
        outArray[2, :, :, :, 0] = sitk.GetArrayFromImage(imgs[2])

    return outArray

def splitTrainAndTest(num_examples, test_perc):
    '''
    Splits a number into training and test randomly
    :param num_examples: int of the number of examples
    :param test_perc: int of the percentage desired for testing
    :return:
    '''
    all_samples_idx = np.arange(num_examples)
    np.random.shuffle(all_samples_idx)
    test_examples = int(np.ceil(num_examples*test_perc))
    # Train and validation indexes
    train_val_idx = all_samples_idx[0:len(all_samples_idx)-test_examples]
    test_idx = all_samples_idx[len(all_samples_idx)-test_examples:len(all_samples_idx)]

    return [train_val_idx, test_idx]

def splitTrainValidationAndTest(num_examples, val_perc, test_perc):
    '''
    Splits a number into training, validation, and test randomly
    :param num_examples: int of the number of examples
    :param val_perc: int of the percentage desired for validation
    :param test_perc: int of the percentage desired for testing
    :return:
    '''
    all_samples_idx = np.arange(num_examples)
    np.random.shuffle(all_samples_idx)
    test_examples = int(np.ceil(num_examples*test_perc))
    val_examples = int(np.ceil(num_examples*val_perc))
    # Train and validation indexes
    train_idx = all_samples_idx[0:len(all_samples_idx)-test_examples-val_examples]
    val_idx = all_samples_idx[len(all_samples_idx)-test_examples-val_examples:len(all_samples_idx)-test_examples]
    test_idx = all_samples_idx[len(all_samples_idx)-test_examples:]

    return [train_idx, val_idx, test_idx]

def createFolder(folder):
    if not(os.path.exists(folder)):
        os.makedirs(folder)

def shift3D(A, size, axis):
    dims = A.shape
    if axis==0:
        if size > 0:
            B = np.lib.pad(A[:dims[1]-size,:,:], ((size, 0), (0, 0), (0,0)), 'edge')
        else:
            B = np.lib.pad(A[-size:, :,:], ((0, -size), (0, 0), (0,0)), 'edge')
    if axis==1:
        if size > 0:
            B = np.lib.pad(A[:,:dims[1]-size, :], ((0, 0), (size, 0), (0,0)), 'edge')
        else:
            B = np.lib.pad(A[:,-size:, :], ((0, 0), (0, -size), (0,0)), 'edge')
    if axis==2:
        if size > 0:
            B = np.lib.pad(A[:,:,:dims[1]-size], ((0, 0), (0,0), (size, 0)), 'edge')
        else:
            B = np.lib.pad(A[:,:,-size: ], ((0, 0), (0,0), (0, -size)), 'edge')
    return B

def makeSphere(size, r):
    corAll = np.arange(size)
    X, Y, Z = np.meshgrid(corAll,corAll,corAll)
    center = np.floor(size/2)
    vals = np.square(X-center) + np.square(Y-center) + np.square(Z-center)
    A = vals < (r*r)
    return A

def copyItkImage(itk_src, np_arr):
    out_itk = sitk.GetImageFromArray(np_arr)
    out_itk.SetOrigin(itk_src.GetOrigin())
    out_itk.SetDirection(itk_src.GetDirection())
    out_itk.SetSpacing(itk_src.GetSpacing())
    return out_itk

# if __name__ == '__main__':
    # Random image

    # # ----- Testing shifting --------
    # simg = 10
    # A = np.random.random((simg, simg, simg))
    # axis = 0
    # shift = -10

    # A[:,0:20:2,:] = 10

    # plt.imshow(A[:,:,-1])
    # plt.show()
    # B = shift3D(A, shift, axis)
    # plt.imshow(B[:,:,-1])
    # plt.show()

