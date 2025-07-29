import tensorflow as tf
import keras
from matplotlib import pyplot as plt

train_gen = tf.keras.preprocessing.image_dataset_from_directory(
  directory="../ogdata",
  labels='inferred',
  label_mode='int',
  color_mode='rgb',
  batch_size=32,
  image_size=(224, 224),
  seed=2,
  validation_split=0.2,
  subset='training'
)

val_gen = tf.keras.preprocessing.image_dataset_from_directory(
  directory="../ogdata",
  labels='inferred',
  label_mode='int',
  color_mode='rgb',
  batch_size=32,
  image_size=(224, 224),
  seed=2,
  validation_split=0.2,
  subset='validation'
)

clwt = {0 : 2.23,
        1 : 1.00}

data_aug = keras.models.Sequential([
  keras.layers.RandomFlip('horizontal'),
  keras.layers.RandomRotation(0.1),
  keras.layers.RandomZoom(0.1),
])

base = keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base.trainable = True
for layer in base.layers[:-30]:
  layer.trainable = False

model = keras.Sequential([
  keras.layers.Input(shape=(224, 224, 3)),
  data_aug,
  base,
  keras.layers.GlobalAveragePooling2D(),
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dropout(0.4),
  keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

callbk1 = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
callbk2 = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                   patience=3, min_lr=1e-6)


bread = model.fit(train_gen, epochs=20, class_weight=clwt, validation_data=val_gen, callbacks=[callbk1, callbk2])

model.save('breadv2-xray-test.keras')

plt.plot(bread.history['accuracy'], label="training accuracy")
plt.plot(bread.history['val_accuracy'], label="validation accuracy")
plt.xlabel('No. of epochs')
plt.ylabel('Accuracy')
plt.title('Model Training Summary')
plt.legend()
plt.show()
