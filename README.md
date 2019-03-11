# Segmentation of prostate and prostate zones using Deep Learning: a multi-MRI vendor analysis
This repository contains a test case for one of our recent projects which consist of an automatic
segmentation algorithm for the prostate and its peripheral zone (PZ) using a 3D Convolutional Neural Network (CNN). 

The details of our algorithm will soon be published and we will add a link in here. 

With this program **six** different CNN models can be tested in **two** sample datasets. The two datasets contain MR images from 
different vendors: Siemens and GE. And the models are split in 3 for **prostate** segmentation and 3 for **PZ** segmentation. The
difference between the models is the training dataset used to build them, not the CNN architecture. 

The architecture used to train our models is the following:
![alt text](https://github.com/olmozavala/Prostate_and_PZ_DL_Segmentation_Code/raw/master/images/NN.png "CNN model")

## Install

**Important!!** This code uses f-strings, which are available since Python 3.6. If you are using an older version please reformat the `print` statements.

### Conda 

```
conda install -c anaconda tensorflow-gpu 
conda install -c simpleitk simpleitk 
conda install -c simpleitk/label/dev simpleitk
conda install -c conda-forge keras 
```

#### Conda  from requirements.txt file (linux)
`
while read requirement; do conda install --yes $requirement || pip install $requirement; done < requirementsT.txt
`

## Data
The test data is freely available and can be downloaded from this link [data](https://goo.gl/193hqk). In order to edit the 
configuration file as little as possible, I suggest you to copy the data folder at the root folder of this repository.

## Run Test
The organization of the folders is the one below, and its content is self explanatory. 
The file to run the test is inside the *code* folder and is `MakeSegmentation.py`. To configure the run you **must** edit
the file `MainConfig.py` inside *code/config*. The configuration file is well documented and there is not need to re-explain it
here. Just mention that in that file you can configure your input and output folders, which model to test, in which dataset, etc. 

<img src="https://github.com/olmozavala/Prostate_and_PZ_DL_Segmentation_Code/raw/master/images/tree.png" width="230"/>

To test any of the models, first edit the configuration file and then run it with:
```
cd code
python MakeSegmentation.py
```

This program will make a segmentation with the proposed CNN and it will create images showing the ground truth contour and the
predicted contour. It will also compute the Dice Coefficient of the segmentation and it will save it in a CSV file.
Some of the images you should be able to generate with this test case are:


Prostate Segmentation | PZ Segmentation
:---------:|:---------:
![alt text](https://github.com/olmozavala/Prostate_and_PZ_DL_Segmentation_Code/raw/master/images/ex1.png "Prostate segmentation") | ![alt text](https://github.com/olmozavala/Prostate_and_PZ_DL_Segmentation_Code/raw/master/images/ex2.png "PZ segmentation")

