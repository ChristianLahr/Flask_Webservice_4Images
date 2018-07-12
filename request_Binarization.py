# Christian Lahr

import requests
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import base64
import numpy as np

# initialize the Keras REST API endpoint URL along with the input image path
REST_API_URL = "http://localhost:5000/predict"
import sys

IMAGE_PATH = "/Users/Chris/PycharmProjects/Binarization/assets/test_documents/receipt/6. online-receipt-location.jpg"
#IMAGE_PATH = sys.argv[1]

image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

# submit the request
r = requests.post(REST_API_URL, files=payload).json()

# ensure the request was sucessful
if r["success"]:
	print("receive image")
	image_binarized =  np.frombuffer(base64.b64decode(r["image_binarized"]), dtype=r["image_binarized_dtype"])
	print("Client: Shape of receives image:", image_binarized.shape)
	image_binarized = image_binarized.reshape(r["image_binarized_shape"])
	print("Client: Shape of receives image:", image_binarized.shape)

	print("compare the images")
	image = Image.open(IMAGE_PATH)
	f = plt.figure()
	f.add_subplot(1,2, 1)
	plt.imshow(image)
	f.add_subplot(1,2, 2)
	plt.imshow(np.round(image_binarized)*255,cmap='gray')
	plt.show(block=True)

# otherwise, the request failed
else:
	print("Request failed")

