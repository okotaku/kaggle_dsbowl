# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np


def make_df(train_path, test_path, img_size):
    """
    データ読み込み

    inputs
    -----
    train_path(str): 学習用データフォルダパス
    test_path(str): テスト用データフォルダパス
    img_size(int): 画像サイズ

    outputs
    -----
    X_train(np.array): 入力画像
    Y_train(np.array): 出力画像
    X_test(np.array): テスト用入力画像
    sizes_test(list): テスト用画像のオリジナルサイズ
    """
    train_ids = next(os.walk(train_path))[1]
    test_ids = next(os.walk(test_path))[1]
    X_train = np.zeros((len(train_ids), img_size, img_size, 3), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), img_size, img_size, 1), dtype=np.bool)
    for i, id_ in enumerate(train_ids):
        path = train_path + id_
        img = cv2.imread(path + '/images/' + id_ + '.png')
        img = cv2.resize(img, (img_size, img_size))
        X_train[i] = img
        mask = np.zeros((img_size, img_size, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = cv2.imread(path + '/masks/' + mask_file, 0)
            mask_ = cv2.resize(mask_, (img_size, img_size))
            mask_ = mask_[:, :, np.newaxis]
            mask = np.maximum(mask, mask_)
        Y_train[i] = mask
    X_test = np.zeros((len(test_ids), img_size, img_size, 3), dtype=np.uint8)
    sizes_test = []
    for i, id_ in enumerate(test_ids):
        path = test_path + id_
        img = cv2.imread(path + '/images/' + id_ + '.png')
        sizes_test.append([img.shape[0], img.shape[1]])
        img = cv2.resize(img, (img_size, img_size))
        X_test[i] = img

    return X_train, Y_train, X_test, sizes_test
