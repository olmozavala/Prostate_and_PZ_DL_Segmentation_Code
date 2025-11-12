"""
3D U-Net models for medical image segmentation.

This module provides implementations of 3D multi-stream U-Net architectures
for segmenting prostate and peripheral zone (PZ) from multi-planar MRI images.
"""
from tensorflow.keras.layers import (
    Input, Conv3D, MaxPooling3D, Conv3DTranspose, 
    BatchNormalization, Dropout, concatenate
)
from tensorflow.keras.models import Model
from typing import List, Tuple, Union, Any


MODEL_NAMES = {
    "3ds": "3D_SingleStream",
    "3dm": "3D_MultiStream",
    "3dmorig": "3D_MultiStreamOriginal",
    "3ddropout": "3D_UsingDropout"
}


def readProperModel(model_name: str, img_size: int) -> Model:
    """
    Creates and returns the appropriate model architecture.
    
    Args:
        model_name: Name of the model architecture ('3dm' or '3ddropout')
        img_size: Size of input images (typically 168)
        
    Returns:
        Compiled Keras model
        
    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name == "3dm":
        model = getModel_3D_Multi([img_size, img_size, img_size], 'sigmoid')
    elif model_name == "3ddropout":
        model = getModel_3D_Multi_Dropout([img_size, img_size, img_size], 'sigmoid')
    else:
        raise ValueError(f"Unknown model name: {model_name}. "
                        f"Supported models: {list(MODEL_NAMES.keys())}")
    return model


def getModel_3D_Multi(imgs_dims: List[int], last_layer: str = 'sigmoid') -> Model:
    """
    Creates a 3D multi-stream U-Net model for multi-planar segmentation.
    
    This architecture takes three orthogonal views (transversal, sagittal, coronal)
    as separate inputs and merges them in the bottleneck layer.
    
    Args:
        imgs_dims: List of image dimensions [width, height, depth]
        last_layer: Activation function for the output layer ('sigmoid' or 'softmax')
        
    Returns:
        Compiled Keras model with three inputs (tra, sag, cor) and one output
    """
    filter_factor = 1
    w, h, d = imgs_dims
    
    # ========== Transversal branch ==========
    inputs_tra = Input((w, h, d, 1))
    conv1_tra = Conv3D(8 * filter_factor, (3, 3, 3), activation='relu', padding='same')(inputs_tra)
    conv1_tra = Conv3D(16 * filter_factor, (3, 3, 3), activation='relu', padding='same')(conv1_tra)
    pool1_tra = MaxPooling3D(pool_size=(2, 2, 2))(conv1_tra)
    
    conv2_tra = Conv3D(16 * filter_factor, (3, 3, 3), activation='relu', padding='same')(pool1_tra)
    conv2_tra = Conv3D(32 * filter_factor, (3, 3, 3), activation='relu', padding='same')(conv2_tra)
    pool2_tra = MaxPooling3D(pool_size=(2, 2, 2))(conv2_tra)
    
    conv3_tra = Conv3D(32 * filter_factor, (3, 3, 3), activation='relu', padding='same')(pool2_tra)
    conv3_tra = Conv3D(64 * filter_factor, (3, 3, 3), activation='relu', padding='same')(conv3_tra)
    pool3_tra = MaxPooling3D(pool_size=(2, 2, 2))(conv3_tra)
    
    # ========== Coronal branch ==========
    inputs_cor = Input((w, h, d, 1))
    conv1_cor = Conv3D(8 * filter_factor, (3, 3, 3), activation='relu', padding='same')(inputs_cor)
    conv1_cor = Conv3D(16 * filter_factor, (3, 3, 3), activation='relu', padding='same')(conv1_cor)
    pool1_cor = MaxPooling3D(pool_size=(2, 2, 2))(conv1_cor)
    
    conv2_cor = Conv3D(16 * filter_factor, (3, 3, 3), activation='relu', padding='same')(pool1_cor)
    conv2_cor = Conv3D(32 * filter_factor, (3, 3, 3), activation='relu', padding='same')(conv2_cor)
    pool2_cor = MaxPooling3D(pool_size=(2, 2, 2))(conv2_cor)
    
    conv3_cor = Conv3D(32 * filter_factor, (3, 3, 3), activation='relu', padding='same')(pool2_cor)
    conv3_cor = Conv3D(64 * filter_factor, (3, 3, 3), activation='relu', padding='same')(conv3_cor)
    pool3_cor = MaxPooling3D(pool_size=(2, 2, 2))(conv3_cor)
    
    # ========== Sagittal branch ==========
    inputs_sag = Input((w, h, d, 1))
    conv1_sag = Conv3D(8 * filter_factor, (3, 3, 3), activation='relu', padding='same')(inputs_sag)
    conv1_sag = Conv3D(16 * filter_factor, (3, 3, 3), activation='relu', padding='same')(conv1_sag)
    pool1_sag = MaxPooling3D(pool_size=(2, 2, 2))(conv1_sag)
    
    conv2_sag = Conv3D(16 * filter_factor, (3, 3, 3), activation='relu', padding='same')(pool1_sag)
    conv2_sag = Conv3D(32 * filter_factor, (3, 3, 3), activation='relu', padding='same')(conv2_sag)
    pool2_sag = MaxPooling3D(pool_size=(2, 2, 2))(conv2_sag)
    
    conv3_sag = Conv3D(32 * filter_factor, (3, 3, 3), activation='relu', padding='same')(pool2_sag)
    conv3_sag = Conv3D(64 * filter_factor, (3, 3, 3), activation='relu', padding='same')(conv3_sag)
    pool3_sag = MaxPooling3D(pool_size=(2, 2, 2))(conv3_sag)
    
    # ========== Merge branches ==========
    merge = concatenate([pool3_tra, pool3_cor, pool3_sag])
    
    # ========== Bottleneck ==========
    conv4 = Conv3D(128 * filter_factor, (3, 3, 3), activation='relu', padding='same')(merge)
    conv5 = BatchNormalization(axis=4)(conv4)
    conv6 = Conv3D(128 * filter_factor, (3, 3, 3), activation='relu', padding='same')(conv5)
    conv7 = BatchNormalization(axis=4)(conv6)
    up6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same')(conv7)
    
    # ========== Decoder with skip connections ==========
    up6 = concatenate([up6, conv3_tra, conv3_cor, conv3_sag])
    
    conv8 = Conv3D(64 * filter_factor, (3, 3, 3), activation='relu', padding='same')(up6)
    conv9 = BatchNormalization(axis=4)(conv8)
    conv10 = Conv3D(64 * filter_factor, (3, 3, 3), activation='relu', padding='same')(conv9)
    conv11 = BatchNormalization(axis=4)(conv10)
    up7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same')(conv11)
    
    up7 = concatenate([up7, conv2_tra, conv2_cor, conv2_sag])
    
    conv12 = Conv3D(32 * filter_factor, (3, 3, 3), activation='relu', padding='same')(up7)
    conv13 = BatchNormalization(axis=4)(conv12)
    conv14 = Conv3D(32 * filter_factor, (3, 3, 3), activation='relu', padding='same')(conv13)
    conv15 = BatchNormalization(axis=4)(conv14)
    up8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same')(conv15)
    
    up8 = concatenate([up8, conv1_tra, conv1_cor, conv1_sag])
    
    # ========== Final layers ==========
    conv16 = Conv3D(16 * filter_factor, (3, 3, 3), activation='relu', padding='same')(up8)
    conv17 = BatchNormalization(axis=4)(conv16)
    conv18 = Conv3D(16 * filter_factor, (3, 3, 3), activation='relu', padding='same')(conv17)
    conv19 = BatchNormalization(axis=4)(conv18)
    
    conv20 = Conv3D(1, (1, 1, 1), activation=last_layer)(conv19)
    
    model = Model(inputs=[inputs_tra, inputs_sag, inputs_cor], outputs=[conv20])
    
    return model


def getModel_3D_Multi_Dropout(imgs_dims: List[int], last_layer: str = 'sigmoid') -> Model:
    """
    Creates a 3D multi-stream U-Net model with dropout regularization.
    
    Similar to getModel_3D_Multi but includes dropout layers to reduce overfitting.
    
    Args:
        imgs_dims: List of image dimensions [width, height, depth]
        last_layer: Activation function for the output layer ('sigmoid' or 'softmax')
        
    Returns:
        Compiled Keras model with three inputs (tra, sag, cor) and one output
    """
    filter_factor = 1
    w, h, d = imgs_dims
    dropout_rate = 0.2
    
    # ========== Transversal branch ==========
    inputs_tra = Input((w, h, d, 1))
    conv1_tra = Conv3D(8 * filter_factor, (3, 3, 3), activation='relu', padding='same')(inputs_tra)
    conv1_tra = Conv3D(16 * filter_factor, (3, 3, 3), activation='relu', padding='same')(conv1_tra)
    pool1_tra = MaxPooling3D(pool_size=(2, 2, 2))(conv1_tra)
    
    conv2_tra = Conv3D(16 * filter_factor, (3, 3, 3), activation='relu', padding='same')(pool1_tra)
    conv2_tra = Conv3D(32 * filter_factor, (3, 3, 3), activation='relu', padding='same')(conv2_tra)
    pool2_tra = MaxPooling3D(pool_size=(2, 2, 2))(conv2_tra)
    
    conv3_tra = Conv3D(32 * filter_factor, (3, 3, 3), activation='relu', padding='same')(pool2_tra)
    conv3_tra = Conv3D(64 * filter_factor, (3, 3, 3), activation='relu', padding='same')(conv3_tra)
    pool3_tra = MaxPooling3D(pool_size=(2, 2, 2))(conv3_tra)
    
    # ========== Coronal branch ==========
    inputs_cor = Input((w, h, d, 1))
    conv1_cor = Conv3D(8 * filter_factor, (3, 3, 3), activation='relu', padding='same')(inputs_cor)
    conv1_cor = Conv3D(16 * filter_factor, (3, 3, 3), activation='relu', padding='same')(conv1_cor)
    pool1_cor = MaxPooling3D(pool_size=(2, 2, 2))(conv1_cor)
    
    conv2_cor = Conv3D(16 * filter_factor, (3, 3, 3), activation='relu', padding='same')(pool1_cor)
    conv2_cor = Conv3D(32 * filter_factor, (3, 3, 3), activation='relu', padding='same')(conv2_cor)
    pool2_cor = MaxPooling3D(pool_size=(2, 2, 2))(conv2_cor)
    
    conv3_cor = Conv3D(32 * filter_factor, (3, 3, 3), activation='relu', padding='same')(pool2_cor)
    conv3_cor = Conv3D(64 * filter_factor, (3, 3, 3), activation='relu', padding='same')(conv3_cor)
    pool3_cor = MaxPooling3D(pool_size=(2, 2, 2))(conv3_cor)
    
    # ========== Sagittal branch ==========
    inputs_sag = Input((w, h, d, 1))
    conv1_sag = Conv3D(8 * filter_factor, (3, 3, 3), activation='relu', padding='same')(inputs_sag)
    conv1_sag = Conv3D(16 * filter_factor, (3, 3, 3), activation='relu', padding='same')(conv1_sag)
    pool1_sag = MaxPooling3D(pool_size=(2, 2, 2))(conv1_sag)
    
    conv2_sag = Conv3D(16 * filter_factor, (3, 3, 3), activation='relu', padding='same')(pool1_sag)
    conv2_sag = Conv3D(32 * filter_factor, (3, 3, 3), activation='relu', padding='same')(conv2_sag)
    pool2_sag = MaxPooling3D(pool_size=(2, 2, 2))(conv2_sag)
    
    conv3_sag = Conv3D(32 * filter_factor, (3, 3, 3), activation='relu', padding='same')(pool2_sag)
    conv3_sag = Conv3D(64 * filter_factor, (3, 3, 3), activation='relu', padding='same')(conv3_sag)
    pool3_sag = MaxPooling3D(pool_size=(2, 2, 2))(conv3_sag)
    
    # ========== Merge branches ==========
    merge = concatenate([pool3_tra, pool3_cor, pool3_sag])
    
    # ========== Bottleneck with dropout ==========
    conv4 = Conv3D(128 * filter_factor, (3, 3, 3), activation='relu', padding='same')(merge)
    conv5 = BatchNormalization(axis=4)(conv4)
    conv5 = Dropout(rate=dropout_rate)(conv5)
    conv6 = Conv3D(128 * filter_factor, (3, 3, 3), activation='relu', padding='same')(conv5)
    conv7 = BatchNormalization(axis=4)(conv6)
    conv7 = Dropout(rate=dropout_rate)(conv7)
    up6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same')(conv7)
    
    # ========== Decoder with skip connections and dropout ==========
    up6 = concatenate([up6, conv3_tra, conv3_cor, conv3_sag])
    
    conv8 = Conv3D(64 * filter_factor, (3, 3, 3), activation='relu', padding='same')(up6)
    conv9 = BatchNormalization(axis=4)(conv8)
    conv9 = Dropout(rate=dropout_rate)(conv9)
    conv10 = Conv3D(64 * filter_factor, (3, 3, 3), activation='relu', padding='same')(conv9)
    conv11 = BatchNormalization(axis=4)(conv10)
    conv11 = Dropout(rate=dropout_rate)(conv11)
    up7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same')(conv11)
    
    up7 = concatenate([up7, conv2_tra, conv2_cor, conv2_sag])
    
    conv12 = Conv3D(32 * filter_factor, (3, 3, 3), activation='relu', padding='same')(up7)
    conv13 = BatchNormalization(axis=4)(conv12)
    conv13 = Dropout(rate=dropout_rate)(conv13)
    conv14 = Conv3D(32 * filter_factor, (3, 3, 3), activation='relu', padding='same')(conv13)
    conv15 = BatchNormalization(axis=4)(conv14)
    conv15 = Dropout(rate=dropout_rate)(conv15)
    up8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same')(conv15)
    
    up8 = concatenate([up8, conv1_tra, conv1_cor, conv1_sag])
    
    # ========== Final layers with dropout ==========
    conv16 = Conv3D(16 * filter_factor, (3, 3, 3), activation='relu', padding='same')(up8)
    conv17 = BatchNormalization(axis=4)(conv16)
    conv17 = Dropout(rate=dropout_rate)(conv17)
    conv18 = Conv3D(16 * filter_factor, (3, 3, 3), activation='relu', padding='same')(conv17)
    conv19 = BatchNormalization(axis=4)(conv18)
    conv19 = Dropout(rate=dropout_rate)(conv19)
    
    conv20 = Conv3D(1, (1, 1, 1), activation=last_layer)(conv19)
    
    model = Model(inputs=[inputs_tra, inputs_sag, inputs_cor], outputs=[conv20])
    
    return model
