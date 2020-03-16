import numpy as np

from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.preprocessing import image


MODEL_PATH = "resnet50_best.h5"
IMAGE_SIZE = (224, 224)
IMAGE_PATH = "../cropped_faces/resnet_data/train/fake/1_2.png"


def predict(img_path, model):
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    return preds


def main():
    print("Loading model:", MODEL_PATH)
    #model = load_model(MODEL_PATH)
    print("Loaded model")
    model = 0
    preds = predict(IMAGE_PATH, model)
    print(preds)


main()
