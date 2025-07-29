import tensorflow as tf
import numpy as np
import keras
import os
import matplotlib.pyplot as plt

model = keras.models.load_model("bread-xray-test.keras")

img_path='/home/bread/projects/xrayclass/nofrac.jpeg'
img = keras.preprocessing.image.load_img(img_path, target_size=(256, 256))
img_array = keras.preprocessing.image.img_to_array(img)
img_array = img_array/255.0
img_array = np.expand_dims(img_array, axis=0)


img2_path='/home/bread/projects/xrayclass/nofrac2.jpg'
img2 = keras.preprocessing.image.load_img(img2_path, target_size=(256, 256))
img2_array = keras.preprocessing.image.img_to_array(img2)
img2_array = img2_array/255.0
img2_array = np.expand_dims(img2_array, axis=0)


img3_path='/home/bread/projects/xrayclass/frac.jpg'
img3 = keras.preprocessing.image.load_img(img3_path, target_size=(256, 256))
img3_array = keras.preprocessing.image.img_to_array(img3)
img3_array = img3_array/255.0
img3_array = np.expand_dims(img3_array, axis=0)

img4_path='/home/bread/projects/xrayclass/frac2.jpg'
img4 = keras.preprocessing.image.load_img(img4_path, target_size=(256, 256))
img4_array = keras.preprocessing.image.img_to_array(img4)
img3_array = img3_array/255.0
img4_array = np.expand_dims(img4_array, axis=0)



#img1, img2 = notfrac
#img3, img4 = frac

prediction = model.predict(img_array)[0][0]
label=''
if prediction<0.5:
  label='Fracture'
else:
  label='Normal'

prediction2 = model.predict(img2_array)[0][0]
label2=''
if prediction2<0.5:
  label2='Fracture'
else:
  label2='Normal'

prediction3 = model.predict(img3_array)[0][0]
label3=''
if prediction3<0.5:
  label3='Fracture'
else:
  label3='Normal'

prediction4 = model.predict(img4_array)[0][0]
label4=''
if prediction4<0.5:
  label4='Fracture'
else:
  label4='Normal'

print("acc to the model the frac image is :", label)
print(f"Raw model output for frac (sigmoid score): {prediction}")

print("acc to the model the second frac image is :", label2)
print(f"Raw model output for second frac (sigmoid score): {prediction2}")

print("acc to the model the nonfrac image is :", label2)
print(f"Raw model output for nonfrac (sigmoid score): {prediction2}")

print("acc to the model the second nonfrac image is :", label2)
print(f"Raw model output for second nonfrac (sigmoid score): {prediction2}")
