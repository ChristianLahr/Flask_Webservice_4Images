# Christian Lahr
#
# Send requests to the binarization server.
#
# Arguments:
# 1: String: Mode ('path', 'image')
# 2: String: Path
# 3: Boolean: Save result
# 4: Boolean: compare images
#
# Mode 1: folder
# Binarize all images in the folder an save them if desired in the subfolder "binarized_images".
# python python request_Binarization_v2.py 'folder' 'path/to/the/folder/' True
#
# Mode 2: image
# Binarize an image and save it if desired.
# python python request_Binarization_v2.py 'image' 'path/to/the/image/' True


import requests
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import base64
import numpy as np
import sys
import os

REST_API_URL = "http://localhost:5000/predict"

LoadType = sys.argv[1]
PATH = sys.argv[2]
save_images = sys.argv[3]
compare_images = sys.argv[4]

def image_request(IMAGE_PATH, save_images, destination_path, compare_images):
    image = open(IMAGE_PATH, "rb").read()
    payload = {"image": image}

    # submit the request
    r = requests.post(REST_API_URL, files=payload).json()

    # ensure the request was sucessful
    if r["success"]:
        print("Client: Receive image")
        image_binarized =  np.frombuffer(base64.b64decode(r["image_binarized"]), dtype=r["image_binarized_dtype"])
        image_binarized = image_binarized.reshape(r["image_binarized_shape"])

        if compare_images:
            print("Client: Compare the images")
            image = Image.open(IMAGE_PATH)
            f = plt.figure()
            f.add_subplot(1,2, 1)
            plt.imshow(image)
            f.add_subplot(1,2, 2)
            plt.imshow(np.round(image_binarized)*255,cmap='gray')
            plt.show(block=False)

        if save_images:
            print("Client: Save the image")
            cv2.imwrite(destination_path, np.round(image_binarized*255))

    # otherwise, the request failed
    else:
        print("Request failed")


if LoadType == 'image':
    filename_path, extension = os.path.splitext(PATH)
    destination_path = os.path.join(filename_path + "_binarized" + extension)
    print(PATH)
    print(destination_path)
    image_request(PATH, save_images, destination_path, compare_images)

elif LoadType == 'folder':
    VALID_IMAGES = [".jpg", ".png", ".PNG", ".JPG"]
    n = 0
    for file in os.listdir(PATH):
        extension = os.path.splitext(file)[1]
        if extension.lower() in VALID_IMAGES:
            n += 1
    print("found", n, "images")
    i = 1
    destination_dir = os.path.join(PATH, "binarized_images")
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    for file in os.listdir(PATH):
        filename, extension = os.path.splitext(file)
        if extension.lower() in VALID_IMAGES:
            print("(%i/%i) binarize: %s" %(i, n, file))
            destination_path = os.path.join(destination_dir, filename + "_binarized" + extension)
            image_request(os.path.join(PATH, file), save_images, destination_path, compare_images)
            i += 1
else:
    print("specify correct mode")
