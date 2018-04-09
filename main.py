import cv2
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint

from generator import generator
from make_df import make_df
from model.unet import Unet
from utils import bce_dice_loss
from utils import recall_score
from utils import precision_score
from utils import rocauc_score
from utils import prob_to_rles


if __name__ == "__main__":
    img_size = 256
    batch_size = 32
    train_path = './data/stage1_train/'
    test_path = './data/stage1_test/'

    X_train, Y_train, X_test, sizes_test = make_df(train_path, test_path,
                                                   img_size)
    xtr, xval, ytr, yval = train_test_split(X_train, Y_train, test_size=0.1,
                                            random_state=7)
    train_generator, val_generator = generator(xtr, xval, ytr, yval, batch_size)

    model = Unet(img_size)
    model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[bce_dice_loss,
     recall_score, precision_score, rocauc_score])
    ckpt = ModelCheckpoint('.model.hdf5', save_best_only=True,
                           monitor='val_rocauc_score', mode='max')

    model.fit_generator(train_generator, steps_per_epoch=len(xtr)/6,
                        epochs=50, validation_data=val_generator,
                        validation_steps=len(xval)/batch_size,
                        callbacks=[ckpt])
    model.load_weights('.model.hdf5')

    preds_test = model.predict(X_test, verbose=1)

    preds_test_upsampled = []
    for i in range(len(preds_test)):
        preds_test_upsampled.append(cv2.resize(preds_test[i],
                                    (sizes_test[i][1], sizes_test[i][0])))

    test_ids = next(os.walk(test_path))[1]
    new_test_ids = []
    rles = []
    for n, id_ in enumerate(test_ids):
        rle = list(prob_to_rles(preds_test_upsampled[n]))
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))
    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv('sub.csv', index=False)
