import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
        help="input image to be processed")
ap.add_argument("-t", "--type", type=str, default="gauss",
        help="specify noise type to add to image")
ap.add_argument("-r", "--ratio", type=float, default=1.0,
        help="effect ratio of noise to image")
args = vars(ap.parse_args())

def gauss_noise(image, ratio = 1):
    gauss = np.random.normal(0, 1, image.shape)
    gauss = gauss * ratio
    gauss = gauss.reshape(image.shape).astype('uint8')
    image_gauss = image + image * gauss
    return image_gauss

img = cv2.imread(args["image"])
if args["type"] == "gauss":
    img_gauss = gauss_noise(img, args["ratio"])
    export_file = "{}_gauss.png".format(args["image"])
    cv2.imwrite(export_file, img_gauss)
    cv2.imshow('img', img_gauss)
    cv2.waitKey(0)
