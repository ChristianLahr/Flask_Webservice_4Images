import requests

# initialize the Keras REST API endpoint URL along with the input image path
KERAS_REST_API_URL = "http://localhost:5000/predict"
import sys
IMAGE_PATH = sys.argv[1]

image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

# submit the request
r = requests.post(KERAS_REST_API_URL, files=payload).json()

# ensure the request was sucessful
if r["success"]:
	for (i, result) in enumerate(r["predictions"]):
		print("{}. {}: {:.4f}".format(i + 1, result["label"],
			result["probability"]))

# otherwise, the request failed
else:
	print("Request failed")