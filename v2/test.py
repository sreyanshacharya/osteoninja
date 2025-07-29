import tensorflow as tf
import numpy as np
import keras

model = keras.models.load_model("breadv2-xray-test.keras")

def preprocess(path):
    img = keras.preprocessing.image.load_img(path, target_size=(224, 224))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

# load & preprocess
img1 = preprocess('../realimages/nofrac.jpeg')
img2 = preprocess('../realimages/nofrac2.jpg')
img3 = preprocess('../realimages/frac.jpg')
img4 = preprocess('../realimages/frac2.jpg')

# predict
pred1 = model.predict(img1)[0][0]
pred2 = model.predict(img2)[0][0]
pred3 = model.predict(img3)[0][0]
pred4 = model.predict(img4)[0][0]

def interpret(pred):
    return 'Fracture' if pred < 0.5 else 'Normal'

print("acc to the model the nonfrac image is :", interpret(pred1))
print(f"Raw model output for nonfrac: {pred1}\n")

print("acc to the model the second nonfrac image is :", interpret(pred2))
print(f"Raw model output for second nonfrac: {pred2}\n")

print("acc to the model the frac image is :", interpret(pred3))
print(f"Raw model output for frac: {pred3}\n")

print("acc to the model the second frac image is :", interpret(pred4))
print(f"Raw model output for second frac: {pred4}")
