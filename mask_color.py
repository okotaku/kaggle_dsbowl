import cv2
import glob
import random
import colorsys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline


def rleToMask(rleString,height,width):
    rows,cols = height,width
    rleNumbers = [int(numstring) for numstring in rleString.split(' ')]
    rlePairs = np.array(rleNumbers).reshape(-1,2)
    img = np.zeros(rows*cols,dtype=np.uint8)
    for index,length in rlePairs:
        index -= 1
        img[index:index+length] = 255
    img = img.reshape(cols,rows)
    img = img.T
    return img

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 255,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = (list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv)))
    random.shuffle(colors)
    return colors


if __name__ == "__main__":
    df = pd.read_csv("sub.csv")
    
    path = glob.glob("./data/stage2_test_final/*")
    for p in path:
        image_id = p.replace("./data/stage2_test_final/", "")
        image_path = glob.glob(p + "/images/*.png")[0]
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 12))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        h, w, _ = img.shape
        df_ = df[df.ImageId == image_id]
        for d in df_.values:
            color = random_colors(1)[0]
            mask = rleToMask(d[1], h, w)
            img = apply_mask(img, mask, color)
        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.savefig("result/{}.png".format(image_id))
        plt.clf()