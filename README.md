# SegmentationofprostateandprostatezonesusingDeepLearning:amulti-MRIvendoranalysis
This repository contains a test case for one of our recent projects which consist of an automatic
segmentation algorithm for the prostate and its peripheral zone (PZ) using a 3D Convolutional Neural Network (CNN). 

The details of our algorithm will soon be published and we will add a link in here. 

In this test case **six** different CNN models can be tested in two sample datasets. The two datasets contain MR images from 
two different vendors: Siemens and GE. And the models are split in 3 for **prostate** segmentation and 3 for **PZ** segmentation. The
difference between the models is the training dataset used to build them, and not the CNN architecture. 

This is the architecture used to train our models:
![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "CNN model")


## Install

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

## Run Test
The organization of the folders is the one below, and its content is self explanatory. 
The file to run the test is inside the *code* folder and is `MakeSegmentation.py`. To configure the run you **must** edit
the file `MainConfig.py` inside *code/config*. The configuration file is well documented and there is not need to re-explain it
here. Just mention that in that file you can configure your input and output folders, which model to test, in which dataset, etc. 

% Image here

To test any of the models, first edit the configuration file and then run it with:
```
cd code
python MakeSegmentation.py
```

This program will make a segmentation with the proposed CNN and it will create images showing the ground truth contour and the
predicted contour. It will also compute the Dice Coefficient of the segmentation and it will save it in a CSV file and it will also
create a bar plot. 

