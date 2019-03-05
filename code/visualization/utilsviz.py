import SimpleITK as sitk
import pylab
import matplotlib.pyplot as plt
import os
from os.path import join
from os import listdir
import numpy as np
import scipy.misc as misc

colors = ['y', 'r', 'c', 'b', 'g', 'w', 'k', 'y', 'r', 'c', 'b', 'g', 'w', 'k']

view_results = False

def dispImages():
    '''This function is used to deside if we want to plot images or only save them'''
    if view_results:
        plt.show()
    else:
        plt.close()

def plotMultipleBarPlots(tuple_dicts, savefig='', title='', legends=''):
    '''
    Plots the DSC as a bar plot
    :param tuple_dicts: List with multiple dictionaries with keys (case str) and values (DSC)
    :param savefig:
    :param title:
    :return:
    '''
    try:
        tot_dsc = len(tuple_dicts) # Number of Dice coefficients to compare
        tot_ex = len(tuple_dicts[0]) # Number of examples
        plt.figure(figsize=(8*tot_ex*tot_dsc/14,8))

        minval = 100000
        maxval = -100000
        for cur_data in tuple_dicts:
            cur_min = min(cur_data.values())
            cur_max = max(cur_data.values())
            if cur_min < minval:
                minval = cur_min
            if cur_max > maxval:
                maxval= cur_max

        for ii, cur_data in enumerate(tuple_dicts):
            if legends != '':
                plt.bar(np.arange(ii,tot_dsc*tot_ex+ii,tot_dsc), tuple_dicts[ii].values(),
                        tick_label=list(tuple_dicts[ii].keys()), align='edge', label=legends[ii])
            else:
                plt.bar(np.arange(ii,tot_dsc*tot_ex+ii,tot_dsc), tuple_dicts[ii].values(),
                        tick_label=list(tuple_dicts[ii].keys()), align='edge')

        plt.xticks(np.arange(0,tot_dsc*tot_ex,tot_dsc), list(tuple_dicts[0].keys()), rotation=20)
        plt.ylim([minval-.1, max(maxval+.1,1)])
        plt.xlim([-1,tot_dsc*tot_ex*1.01])
        plt.legend(loc='best')
        plt.grid()
        if title != '':
            plt.title(title)
        if savefig != '':
            plt.savefig(savefig, bbox_inches='tight')
        dispImages()
    except Exception as e:
        print("----- Not able to make BAR plot for multiple DSC: ", e)


def plotDSC(dsc_scores, savefig='', title=''):
    '''
    Plots the DSC as a bar plot
    :param dsc_scores: Dictionary with keys (case str) and values (DSC)
    :param savefig:
    :param title:
    :return:
    '''
    plt.figure(figsize=(8*(len(dsc_scores)/10),8))
    plt.bar(range(len(dsc_scores)), dsc_scores.values(), tick_label=list(dsc_scores.keys()), align='edge')
    plt.xticks(rotation=28)
    plt.ylim([.5, 1])
    plt.grid()
    if title != '':
        plt.title(title)
    if savefig != '':
        plt.savefig(savefig)
    dispImages()


def drawSeriesItk(img, slices='all', title='', contours=[], savefig='', labels=[], draw_only_contours=True):
    '''

    :param img:
    :param slices: slices can be a string or an array of indexes
    :param title:
    :return:
    '''

    numpy_img = sitk.GetArrayViewFromImage(img)
    ctrs = []
    if len(contours) > 0:
        for contour in contours:
            ctrs.append( sitk.GetArrayViewFromImage(contour) )

    if isinstance(slices,str):
        if slices == 'all':
            slices = range(numpy_img.shape[0])
        if slices == 'middle':
            slices = [ int(np.ceil(img.GetSize()[2]/2)) ]

    drawSlicesNumpy(numpy_img, slices, title, ctrs, savefig, labels, img.GetSize(), draw_only_contours=draw_only_contours)

def drawMultipleSeriesItk(imgs, slices='all', title='', contours=[], savefig='', labels=[], draw_only_contours=True,
                          plane='ax', subtitles=[]):
    numpy_imgs = []
    for img in imgs:
        numpy_imgs.append(sitk.GetArrayViewFromImage(img))

    ctrs = []
    if len(contours) > 0:
        for contour in contours:
            ctrs.append( sitk.GetArrayViewFromImage(contour) )

    if isinstance(slices,str):
        if slices == 'all':
            slices = range(numpy_imgs[0].shape[0])
        if slices == 'middle':
            slices = [ int(np.ceil(imgs[0].GetSize()[2]/2)) ]

    drawMultipleSlices(numpy_imgs, slices, title, ctrs, savefig, labels, draw_only_contours=draw_only_contours,
                       plane=plane, subtitles=subtitles)

def drawMultipleSlices(itk_imgs, slices=['middle'], title='', contours=[], savefig='', labels=[''],
                       colorbar=False, plane='ax', draw_only_contours=True, subtitles=[]):

    totImgs = len(itk_imgs)
    draw_slice = True

    for slice in slices:

        if len(contours) == 0: #If there are no contours, then we always draw the image
            draw_slice = True
        else:
            # Only draw slices where there is at least one contour
            if draw_only_contours:
                draw_slice = False
                for cc in contours:
                    if np.any(np.sum(getProperPlane(cc, plane, slice)) > 0): # Avoid black slices
                        draw_slice = True
                    break # We do not need to verify the others

        if draw_slice:
            fig, ax = plt.subplots(1,totImgs, squeeze=True, figsize=(8*totImgs,8))
            for ii,numpy_img in enumerate(itk_imgs):
                if totImgs == 1: # Special case when we have only one image
                    curax = ax
                else:
                    curax = ax[ii]

                curax.axis('off')
                imres = curax.imshow(getProperPlane(numpy_img,plane,slice), cmap='gray')
                if len(subtitles) > 0: # Adds subtitles into the image
                    curax.set_title(subtitles[ii], fontsize=20)

                if colorbar:
                    plt.colorbar(imres,ax=curax)
                if len(contours) > 0:
                    for idx, cc in enumerate(contours):
                        CS = curax.contour(getProperPlane(cc,plane,slice), colors=colors[idx%len(colors)], linewidths=.4)
                        if len(labels) > 0:
                            curax.clabel(CS, inline=1, fontsize=0)
                            CS.collections[0].set_label(labels[idx])

                    if len(labels) > 0:
                        curax.legend(loc='upper right', framealpha=1, prop={'size':15})

            if title != '': # Only draws a title if is received
                fig.suptitle(title, fontsize=20)

            if savefig != '':
                pylab.savefig('{}_{num:03d}.png'.format(savefig,num=slice), bbox_inches='tight')

            dispImages()

def drawSlicesNumpy(numpy_img, slices, title, ctrs, savefig, labels, imgsize, plane='ax', draw_only_contours=True):

    draw_slice = True
    fig = plt.figure(frameon=False)
    for slice in slices:

        # In this case we will only draw slices where there are contours
        if draw_only_contours:
            draw_slice = False
            for cc in ctrs:
                if np.any(np.sum(getProperPlane(cc, plane, slice)) > 0): # Avoid black slices
                    draw_slice = True
                    break # We do not need to verify the others

        if draw_slice:
            plt.imshow(numpy_img[slice,:,:], cmap='gray')

            if len(ctrs) > 0:
                for idx, cc in enumerate(ctrs):
                    CS = plt.contour(cc[slice, :, :], colors=colors[idx], linewidths=.4)
                    if len(labels) > 0:
                        plt.clabel(CS, inline=1, fontsize=0)
                        CS.collections[0].set_label(labels[idx])

                if len(labels) > 0:
                    plt.legend(loc='upper right', framealpha=1, prop={'size':12})

            plt.title('{} {} slice:{}'.format(title, imgsize, slice))
            plt.axis('off')
            if savefig != '':
                pylab.savefig('{}_{num:03d}.png'.format(savefig,num=slice),bbox_inches='tight')

            dispImages()

def getProperPlane(arr, plane, slice):
    if slice < arr.shape[getAxisIdx(plane)]: # Avoid index out of bounds for images with different # of slices
        if plane == 'ax':
            return arr[slice,:,:]
        if plane == 'sag':
            return arr[:,:,slice]
        if plane == 'cor':
            return arr[:,slice,:]
    else:
        return -1

def getAxisIdx(plane):
    dim_idx = 0
    if plane == 'ax':
        dim_idx = 0
    if plane == 'sag':
        dim_idx = 1
    if plane == 'cor':
        dim_idx = 2
    return dim_idx

def plotMultipleHistograms(all_hist, labels, save_file='', start_at = 0, width=4):

    figure = plt.figure(figsize=(12,8))
    try:
        for ii, c_hist in enumerate(all_hist):
            x = c_hist[1][start_at:-1]
            y = c_hist[0][start_at:]
            plt.bar(x, y, width=width*np.ones(len(x)), alpha=.5, label=labels[ii])

        plt.legend(loc='best')

        if save_file != '':
            pylab.savefig(save_file,bbox_inches='tight')

        dispImages()

    except Exception as e:
        print('---------------------------- Failed {} error: {} ----------------'.format(save_file, e))

def plotHistogramsFromImg(imgs_itk, title='', mode='2d', save_file=''):
    '''
    Plots the histogram and one slice of N number of images
    :param slice: int slice to plot
    :param title: title of the figure
    :param imgs_itk:
    :return:
    '''
    totImgs = len(imgs_itk)
    fig, ax = plt.subplots(2,totImgs, squeeze=True, figsize=(8*totImgs,8))
    for ii, c_img in enumerate(imgs_itk):
        np_img = sitk.GetArrayFromImage(c_img)
        slices = getSlices('middle', [np_img])

        if totImgs > 1:
            t1 = ax[0][ii]
            t2 = ax[1][ii]
        else:
            t1 = ax[0]
            t2 = ax[1]

        if mode == '2d':
            f_img = np_img[slices[0],:,:]
        else:
            f_img = np_img

        t1.hist(f_img.flatten(), 'auto')
        t2.imshow(np_img[slices[0],:,:])

    plt.title(title)
    if save_file != '':
        pylab.savefig(save_file,bbox_inches='tight')

    dispImages()

def getSlices(orig_slices, arr, plane='ax'):

    # Used to decide which axis to use to get the number of slices
    dim_idx = getAxisIdx(plane)
    cur_dim = arr[0].shape[dim_idx]
    if isinstance(orig_slices,np.str): # Deciding if we draw all the images or not
        if orig_slices == 'all':
            slices = range(cur_dim)
        elif orig_slices == 'middle':
            slices = [int(cur_dim/2)]
        elif orig_slices == 'middleThird':
            bottom = int(np.floor(cur_dim/3))
            top = int(np.ceil(cur_dim*(2/3)))
            slices = range(bottom,top)
        else:
            raise Exception(F'The "slices" option is incorrect: {orig_slices}')
    else:
        slices = orig_slices
    return slices
