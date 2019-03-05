import matplotlib.pyplot as plt
import time
from os.path import join, isdir
import pydicom
import numpy as np
from os import walk, listdir
import SimpleITK as sitk

def readImgAndProstate(inFolder, current_folder):
    '''Reads the HR and orginal of transversal and prostate'''
    img_tra_original = sitk.ReadImage(join(inFolder, current_folder, 'img_tra.nrrd'))
    ctr_pro = sitk.ReadImage(join(inFolder, current_folder, 'ctr_pro.nrrd'))

    img_tra_HR = sitk.ReadImage(join(inFolder, current_folder,'hr_tra.nrrd'))
    ctr_pro_HR = sitk.ReadImage(join(inFolder, current_folder, 'hr_ctr_pro.nrrd'))

    startROI = [int(x) for x in np.loadtxt(join(inFolder, current_folder, 'start_ROI.csv'))]
    sizeROI  = [int(x) for x in np.loadtxt(join(inFolder, current_folder, 'size_ROI.csv'))]

    roi_ctr_pro = sitk.ReadImage(join(inFolder, current_folder, 'roi_ctr_pro.nrrd'))
    return [img_tra_original, img_tra_HR, ctr_pro, ctr_pro_HR, roi_ctr_pro, startROI, sizeROI]

def readROI(inFolder, current_folder, type_segmentation):
    ''' Reads the roi files. tra,sag,cor for prostate and pz and tra,bval,adc for lesion'''
    roi_img_tra = sitk.ReadImage(join(inFolder, current_folder, 'roi_tra.nrrd'))

    if type_segmentation == 'PZ' or type_segmentation == 'Prostate':
        roi_img_sag = sitk.ReadImage(join(inFolder, current_folder, 'roi_sag.nrrd'))
        roi_img_cor = sitk.ReadImage(join(inFolder, current_folder, 'roi_cor.nrrd'))
        return [roi_img_tra,roi_img_sag,roi_img_cor]

    if type_segmentation == 'Lesion':
        roi_img_bval = sitk.ReadImage(join(inFolder, current_folder, 'roi_bval.nrrd'))
        roi_img_adc = sitk.ReadImage(join(inFolder, current_folder, 'roi_adc.nrrd'))
        return [roi_img_tra, roi_img_adc, roi_img_bval]

def readPZ(inFolder, current_folder, multistream, img_size):
    '''Obtains the NN input data for case prostate or PZ'''
    ctr_pz_HR = sitk.ReadImage(join(inFolder, current_folder, 'hr_ctr_pz.nrrd'))
    ctr_pz = sitk.ReadImage(join(inFolder, current_folder, 'ctr_pz.nrrd'))
    roi_ctr_pz = sitk.ReadImage(join(inFolder, current_folder, 'roi_ctr_pz.nrrd'))

    return [ctr_pz, ctr_pz_HR, roi_ctr_pz]
