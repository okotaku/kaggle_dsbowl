import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.losses import binary_crossentropy
from skimage.morphology import label


def mean_iou(y_true, y_pred):
    """
    IOUの計算モジュール

    inputs
    -----
    y_true(np.array): 正解画像
    y_pred(np.array): 出力画像

    output
    -----
    float: IOUのスコア
    """
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec))


def dice_coef_(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def bce_dice_loss(y_true, y_pred):
    """
    bce_dice_lossの計算モジュール

    inputs
    -----
    y_true(np.array): 正解画像
    y_pred(np.array): 出力画像

    output
    -----
    float: bce_dice_lossのスコア
    """
    return 0.5 * binary_crossentropy(y_true, y_pred) - dice_coef_(y_true, y_pred)


def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)
