def numpy_dice(y_true, y_pred, smooth=1.0):

    intersection = y_true.flatten()*y_pred.flatten()

    return (2. * intersection.sum() + smooth) / (y_true.sum() + y_pred.sum() + smooth)

