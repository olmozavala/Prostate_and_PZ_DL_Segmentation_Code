from tensorflow.keras.layers import *
from tensorflow.keras.models import *

modelnames = {"3ds":"3D_SingleStream", "3dm":"3D_MultiStream",  "3dmorig":"3D_MultiStreamOriginal", "3ddropout":"3D_UsingDropout" }

def readProperModel(model_name, img_size):
    if model_name == "3dm":
        model = getModel_3D_Multi([img_size,img_size,img_size], 'sigmoid')
    if model_name == "3ddropout":
        model = getModel_3D_Multi_Dropout([img_size,img_size,img_size], 'sigmoid')
    return model

def getModel_3D_Multi(imgs_dims, last_layer='sigmoid'):
    filterFactor = 1
    [w, h, d] = imgs_dims
    # [w, h, d] = [128,128,128]
    # [w, h, d] = [168,168,168]
    #### tra branch #####
    inputs_tra = Input((w, h, d, 1))
    conv1_tra = Conv3D(8*filterFactor, (3, 3, 3), activation='relu', padding='same')(inputs_tra)
    # conv1_tra = BatchNormalization(axis=4)(conv1_tra)
    conv1_tra = Conv3D(16*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv1_tra)
    # conv1_tra = BatchNormalization(axis=4)(conv1_tra)
    pool1_tra = MaxPooling3D(pool_size=(2, 2, 2))(conv1_tra)

    conv2_tra = Conv3D(16*filterFactor, (3, 3, 3), activation='relu', padding='same')(pool1_tra)
    # conv2_tra = BatchNormalization(axis=4)(conv2_tra)
    conv2_tra = Conv3D(32*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv2_tra)
    # conv2_tra = BatchNormalization(axis=4)(conv2_tra)
    pool2_tra = MaxPooling3D(pool_size=(2, 2, 2))(conv2_tra)

    conv3_tra = Conv3D(32*filterFactor, (3, 3, 3), activation='relu', padding='same')(pool2_tra)
    # conv3_tra = BatchNormalization(axis=4)(conv3_tra)
    conv3_tra = Conv3D(64*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv3_tra)
    # conv3_tra = BatchNormalization(axis=4)(conv3_tra)
    pool3_tra = MaxPooling3D(pool_size=(2, 2, 2))(conv3_tra)

    ###### cor branch #####

    inputs_cor = Input((w, h, d, 1))
    conv1_cor = Conv3D(8*filterFactor, (3, 3, 3), activation='relu', padding='same')(inputs_cor)
    # conv1_cor = BatchNormalization(axis=4)(conv1_cor)
    conv1_cor = Conv3D(16*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv1_cor)
    # conv1_cor = BatchNormalization(axis=4)(conv1_cor)
    pool1_cor = MaxPooling3D(pool_size=(2, 2, 2))(conv1_cor)

    conv2_cor = Conv3D(16*filterFactor, (3, 3, 3), activation='relu', padding='same')(pool1_cor)
    # conv2_cor = BatchNormalization(axis=4)(conv2_cor)
    conv2_cor = Conv3D(32*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv2_cor)
    # conv2_cor = BatchNormalization(axis=4)(conv2_cor)
    pool2_cor = MaxPooling3D(pool_size=(2, 2, 2))(conv2_cor)

    conv3_cor = Conv3D(32*filterFactor, (3, 3, 3), activation='relu', padding='same')(pool2_cor)
    # conv3_cor = BatchNormalization(axis=4)(conv3_cor)
    conv3_cor = Conv3D(64*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv3_cor)
    # conv3_cor = BatchNormalization(axis=4)(conv3_cor)
    pool3_cor = MaxPooling3D(pool_size=(2, 2, 2))(conv3_cor)

    ###### sag branch #####

    inputs_sag = Input((w, h, d, 1))
    conv1_sag = Conv3D(8*filterFactor, (3, 3, 3), activation='relu', padding='same')(inputs_sag)
    # conv1_sag = BatchNormalization(axis=4)(conv1_sag)
    conv1_sag = Conv3D(16*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv1_sag)
    # conv1_sag = BatchNormalization(axis=4)(conv1_sag)
    pool1_sag = MaxPooling3D(pool_size=(2, 2, 2))(conv1_sag)

    conv2_sag = Conv3D(16*filterFactor, (3, 3, 3), activation='relu', padding='same')(pool1_sag)
    # conv2_sag = BatchNormalization(axis=4)(conv2_sag)
    conv2_sag = Conv3D(32*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv2_sag)
    # conv2_sag = BatchNormalization(axis=4)(conv2_sag)
    pool2_sag = MaxPooling3D(pool_size=(2, 2, 2))(conv2_sag)

    conv3_sag = Conv3D(32*filterFactor, (3, 3, 3), activation='relu', padding='same')(pool2_sag)
    # conv3_sag = BatchNormalization(axis=4)(conv3_sag)
    conv3_sag = Conv3D(64*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv3_sag)
    # conv3_sag = BatchNormalization(axis=4)(conv3_sag)
    pool3_sag = MaxPooling3D(pool_size=(2, 2, 2))(conv3_sag)

    merge = concatenate([pool3_tra, pool3_cor, pool3_sag])

    # conv4 = Conv3D(192*filterFactor, (3, 3, 3), activation='relu', padding='same')(merge)
    conv4 = Conv3D(128*filterFactor, (3, 3, 3), activation='relu', padding='same')(merge)
    conv5 = BatchNormalization(axis=4)(conv4)
    conv6 = Conv3D(128*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv5)
    conv7 = BatchNormalization(axis=4)(conv6)
    up6 = Conv3DTranspose(128,(2,2,2), strides = (2,2,2), activation = 'relu', padding = 'same' )(conv7)

    up6 = concatenate([up6, conv3_tra, conv3_cor, conv3_sag])

    conv8 = Conv3D(64*filterFactor, (3, 3, 3), activation='relu', padding='same')(up6)
    conv9 = BatchNormalization(axis=4)(conv8)
    conv10 = Conv3D(64*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv9)
    conv11 = BatchNormalization(axis=4)(conv10)
    up7 = Conv3DTranspose(64,(2,2,2), strides = (2,2,2), activation = 'relu', padding = 'same' )(conv11)

    up7 = concatenate([up7, conv2_tra, conv2_cor, conv2_sag])

    conv12 = Conv3D(32*filterFactor, (3, 3, 3), activation='relu', padding='same')(up7)
    conv13 = BatchNormalization(axis=4)(conv12)
    conv14 = Conv3D(32*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv13)
    conv15 = BatchNormalization(axis=4)(conv14)
    up8 = Conv3DTranspose(32,(2,2,2), strides = (2,2,2), activation = 'relu', padding = 'same' )(conv15)

    up8 = concatenate([up8, conv1_tra, conv1_cor, conv1_sag])

    conv16 = Conv3D(16*filterFactor, (3, 3, 3), activation='relu', padding='same')(up8)
    conv17 = BatchNormalization(axis=4)(conv16)
    conv18 = Conv3D(16*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv17)
    conv19 = BatchNormalization(axis=4)(conv18)

    conv20 = Conv3D(1, (1, 1, 1), activation=last_layer)(conv19)

    model = Model(inputs=[inputs_tra, inputs_sag, inputs_cor], outputs=[conv20])

    return model

def getModel_3D_Multi_Dropout(imgs_dims, last_layer='sigmoid'):
    filterFactor = 1
    [w, h, d] = imgs_dims
    # [w, h, d] = [128,128,128]
    # [w, h, d] = [168,168,168]
    #### tra branch #####
    inputs_tra = Input((w, h, d, 1))
    conv1_tra = Conv3D(8*filterFactor, (3, 3, 3), activation='relu', padding='same')(inputs_tra)
    # conv1_tra = BatchNormalization(axis=4)(conv1_tra)
    conv1_tra = Conv3D(16*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv1_tra)
    # conv1_tra = BatchNormalization(axis=4)(conv1_tra)
    pool1_tra = MaxPooling3D(pool_size=(2, 2, 2))(conv1_tra)

    conv2_tra = Conv3D(16*filterFactor, (3, 3, 3), activation='relu', padding='same')(pool1_tra)
    # conv2_tra = BatchNormalization(axis=4)(conv2_tra)
    conv2_tra = Conv3D(32*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv2_tra)
    # conv2_tra = BatchNormalization(axis=4)(conv2_tra)
    pool2_tra = MaxPooling3D(pool_size=(2, 2, 2))(conv2_tra)

    conv3_tra = Conv3D(32*filterFactor, (3, 3, 3), activation='relu', padding='same')(pool2_tra)
    # conv3_tra = BatchNormalization(axis=4)(conv3_tra)
    conv3_tra = Conv3D(64*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv3_tra)
    # conv3_tra = BatchNormalization(axis=4)(conv3_tra)
    pool3_tra = MaxPooling3D(pool_size=(2, 2, 2))(conv3_tra)

    ###### cor branch #####

    inputs_cor = Input((w, h, d, 1))
    conv1_cor = Conv3D(8*filterFactor, (3, 3, 3), activation='relu', padding='same')(inputs_cor)
    # conv1_cor = BatchNormalization(axis=4)(conv1_cor)
    conv1_cor = Conv3D(16*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv1_cor)
    # conv1_cor = BatchNormalization(axis=4)(conv1_cor)
    pool1_cor = MaxPooling3D(pool_size=(2, 2, 2))(conv1_cor)

    conv2_cor = Conv3D(16*filterFactor, (3, 3, 3), activation='relu', padding='same')(pool1_cor)
    # conv2_cor = BatchNormalization(axis=4)(conv2_cor)
    conv2_cor = Conv3D(32*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv2_cor)
    # conv2_cor = BatchNormalization(axis=4)(conv2_cor)
    pool2_cor = MaxPooling3D(pool_size=(2, 2, 2))(conv2_cor)

    conv3_cor = Conv3D(32*filterFactor, (3, 3, 3), activation='relu', padding='same')(pool2_cor)
    # conv3_cor = BatchNormalization(axis=4)(conv3_cor)
    conv3_cor = Conv3D(64*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv3_cor)
    # conv3_cor = BatchNormalization(axis=4)(conv3_cor)
    pool3_cor = MaxPooling3D(pool_size=(2, 2, 2))(conv3_cor)

    ###### sag branch #####

    inputs_sag = Input((w, h, d, 1))
    conv1_sag = Conv3D(8*filterFactor, (3, 3, 3), activation='relu', padding='same')(inputs_sag)
    # conv1_sag = BatchNormalization(axis=4)(conv1_sag)
    conv1_sag = Conv3D(16*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv1_sag)
    # conv1_sag = BatchNormalization(axis=4)(conv1_sag)
    pool1_sag = MaxPooling3D(pool_size=(2, 2, 2))(conv1_sag)

    conv2_sag = Conv3D(16*filterFactor, (3, 3, 3), activation='relu', padding='same')(pool1_sag)
    # conv2_sag = BatchNormalization(axis=4)(conv2_sag)
    conv2_sag = Conv3D(32*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv2_sag)
    # conv2_sag = BatchNormalization(axis=4)(conv2_sag)
    pool2_sag = MaxPooling3D(pool_size=(2, 2, 2))(conv2_sag)

    conv3_sag = Conv3D(32*filterFactor, (3, 3, 3), activation='relu', padding='same')(pool2_sag)
    # conv3_sag = BatchNormalization(axis=4)(conv3_sag)
    conv3_sag = Conv3D(64*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv3_sag)
    # conv3_sag = BatchNormalization(axis=4)(conv3_sag)
    pool3_sag = MaxPooling3D(pool_size=(2, 2, 2))(conv3_sag)

    merge = concatenate([pool3_tra, pool3_cor, pool3_sag])

    # conv4 = Conv3D(192*filterFactor, (3, 3, 3), activation='relu', padding='same')(merge)
    conv4 = Conv3D(128*filterFactor, (3, 3, 3), activation='relu', padding='same')(merge)
    conv5 = BatchNormalization(axis=4)(conv4)
    conv5 = Dropout(rate=0.2)(conv5)
    conv6 = Conv3D(128*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv5)
    conv7 = BatchNormalization(axis=4)(conv6)
    conv7 = Dropout(rate=0.2)(conv7)
    up6 = Conv3DTranspose(128,(2,2,2), strides = (2,2,2), activation = 'relu', padding = 'same' )(conv7)

    up6 = concatenate([up6, conv3_tra, conv3_cor, conv3_sag])

    conv8 = Conv3D(64*filterFactor, (3, 3, 3), activation='relu', padding='same')(up6)
    conv9 = BatchNormalization(axis=4)(conv8)
    conv9 = Dropout(rate=0.2)(conv9)
    conv10 = Conv3D(64*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv9)
    conv11 = BatchNormalization(axis=4)(conv10)
    conv11 = Dropout(rate=0.2)(conv11)
    up7 = Conv3DTranspose(64,(2,2,2), strides = (2,2,2), activation = 'relu', padding = 'same' )(conv11)

    up7 = concatenate([up7, conv2_tra, conv2_cor, conv2_sag])

    conv12 = Conv3D(32*filterFactor, (3, 3, 3), activation='relu', padding='same')(up7)
    conv13 = BatchNormalization(axis=4)(conv12)
    conv13 = Dropout(rate=0.2)(conv13)
    conv14 = Conv3D(32*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv13)
    conv15 = BatchNormalization(axis=4)(conv14)
    conv15 = Dropout(rate=0.2)(conv15)
    up8 = Conv3DTranspose(32,(2,2,2), strides = (2,2,2), activation = 'relu', padding = 'same' )(conv15)

    up8 = concatenate([up8, conv1_tra, conv1_cor, conv1_sag])

    conv16 = Conv3D(16*filterFactor, (3, 3, 3), activation='relu', padding='same')(up8)
    conv17 = BatchNormalization(axis=4)(conv16)
    conv17 = Dropout(rate=0.2)(conv17)
    conv18 = Conv3D(16*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv17)
    conv19 = BatchNormalization(axis=4)(conv18)
    conv19 = Dropout(rate=0.2)(conv19)

    conv20 = Conv3D(1, (1, 1, 1), activation=last_layer)(conv19)

    model = Model(inputs=[inputs_tra, inputs_sag, inputs_cor], outputs=[conv20])

    return model
