# Christian Lahr

# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py


from keras.models import load_model
from keras.preprocessing.image import img_to_array
from utils import get_subwindows, stich_together
import flask
from flask_httpauth import HTTPBasicAuth
import cv2
from PIL import Image
import numpy as np
import io
import base64

import db

HEIGHT = 256
WIDTH = 256
MODEL_PATH = "/Users/Chris/PycharmProjects/Binarization/assets/colin4f/"

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
auth = HTTPBasicAuth()
model = None

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key = True)
    username = db.Column(db.String(32), index = True)
    password_hash = db.Column(db.String(128))

from passlib.apps import custom_app_context as pwd_context

class User(db.Model):
    def hash_password(self, password):
        self.password_hash = pwd_context.encrypt(password)

    def verify_password(self, password):
        return pwd_context.verify(password, self.password_hash)

@auth.verify_password
def verify_password(username, password):
    user = User.query.filter_by(username = username).first()
    if not user or not user.verify_password(password):
        return False
    g.user = user
    return True

def load_model_fct():
    global model
    model = load_model(MODEL_PATH + 'model.hd5f')

def prepare_image(image, HEIGHT, WIDTH):

    # convert image mode
    if image.mode != "RGB":
        image = cv2.cvtColor(img_to_array(image), cv2.COLOR_RGB2GRAY)
    if image.mode != "BGR":
        image = cv2.cvtColor(img_to_array(image), cv2.COLOR_BGR2GRAY)

    image = np.expand_dims(image, axis=2)
    locations, subwindows = get_subwindows(image, height=HEIGHT, width=WIDTH)
    X = np.array(subwindows) / 255

    return X, locations

@app.route("/predict", methods=["POST"])
@auth.login_required
def predict():
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):

            # read the image in PIL format
            print("receive the image")
            image = flask.request.files["image"].read()
            print("open the image")
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            print("prepare the image (create subwindows)")
            X_image, locations = prepare_image(image, HEIGHT = 256, WIDTH = 256) # ResNet need this target dims

            # classify the input image
            print("predict / generate the binarized image")
            pred = model.predict(X_image)
            preds = [p[:, :, 0] for p in pred]
            print("stich subwindows together")
            pred_img = stich_together(locations, subwindows=preds, size=np.array(image).shape[:2])
            image_binarized_shape = pred_img.shape
            image_binarized_dtype = np.array(pred_img).dtype

            print("convert to base64")
            pred_img = base64.b64encode(pred_img)
            pred_img = pred_img.decode('utf-8')

            data["image_binarized"] = pred_img
            data["image_binarized_shape"] = image_binarized_shape
            data["image_binarized_dtype"] = str(image_binarized_dtype)
            data["success"] = True

    # return the data dictionary as a JSON response
    print("convert to json")
    data_json = flask.jsonify(data)
    print("send back")
    return data_json

# if this is the main thread of execution first load the model and then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model_fct()
    print("Certificate required")
    cert_path = '/Users/Chris/PycharmProjects/RestfulWebService/cert/cert.pem'
    key_path = '/Users/Chris/PycharmProjects/RestfulWebService/cert/key.pem'
    app.run(host='0.0.0.0', ssl_context=(cert_path, key_path))
