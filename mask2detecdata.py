import cv2
import os
import numpy as np
import pandas as pd


def make_df(train_path, test_path, img_size):
    train_ids = next(os.walk(train_path))[1]
    test_ids = next(os.walk(test_path))[1]
    X_train = np.zeros((len(train_ids), img_size, img_size, 1), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), img_size, img_size, 1), dtype=np.bool)
    result, result_test = [], []
    for i, id_ in enumerate(train_ids):
        path = train_path + id_
        img = cv2.imread(path + '/images/' + id_ + '.png', 0)
        img_x, img_y = img.shape
        img = cv2.resize(img, (img_size, img_size))
        img = img[:, :, np.newaxis]
        X_train[i] = img
        mask = np.zeros((img_x, img_y, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = cv2.imread(path + '/masks/' + mask_file, 0)
            try:
                xmin, xmax, ymin, ymax = _min_max_checker(mask_)
            except:
                print(path + '/masks/' + mask_file)
                continue
            if i %10 == 0:
                result_test.append([path.replace("./data/", "") + '/images/' + id_ + '.png', xmin, xmax, ymin, ymax, 1])
            else:
                result.append([path.replace("./data/", "") + '/images/' + id_ + '.png', xmin, xmax, ymin, ymax, 1])
            mask_ = mask_[:, :, np.newaxis]
            mask = np.maximum(mask, mask_)
            cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        mask = cv2.resize(mask, (256, 256))
        cv2.imwrite("result/{}.jpg".format(i), mask)
        del mask
    X_test = np.zeros((len(test_ids), img_size, img_size, 1), dtype=np.uint8)
    sizes_test = []
    for i, id_ in enumerate(test_ids):
        path = test_path + id_
        img = cv2.imread(path + '/images/' + id_ + '.png', 0)
        sizes_test.append([img.shape[0], img.shape[1]])
        img = cv2.resize(img, (img_size, img_size))
        img = img[:, :, np.newaxis]
        X_test[i] = img

    return result, result_test

def _min_max_checker(mask_):
    stmask_ = np.hstack(([[0] for i in range(len(mask_))], mask_))
    stmask_ = np.vstack(([0 for i in range(stmask_.shape[1])], stmask_))
    ymin = _pick_min_point(stmask_, axis=0)
    xmin = _pick_min_point(stmask_, axis=1)
    
    reverse_ = mask_[::-1, ::-1]
    streverse_ = np.hstack(([[0] for i in range(len(mask_))], reverse_))
    streverse_ = np.vstack(([0 for i in range(streverse_.shape[1])], streverse_))
    ymax = mask_.shape[0] - _pick_min_point(streverse_, axis=0)
    xmax = mask_.shape[1] - _pick_min_point(streverse_, axis=1)
    
    return xmin, xmax, ymin, ymax
    
def _pick_min_point(mask_, axis):
    points = np.argmax(mask_, axis=axis)
    point = np.min(points[points > 0]) - 1
    return point


if __name__ == "__main__":
    img_size=256
    train_path = './data/stage1_train/'
    test_path = './data/stage1_test/'
    result, result_test = make_df(train_path, test_path, img_size)
    
    pd.DataFrame(result).to_csv("train.csv", index=False, header=None)
    pd.DataFrame(result_test).to_csv("test.csv", index=False, header=None)