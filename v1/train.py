import tensorflow as tf
import keras
import matplotlib
from matplotlib import pyplot as plt

class endtraining(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('val_accuracy')>0.85):
      print("\nintended accuracy reached")
      self.model.stop_training = True

train_gen = tf.keras.preprocessing.image_dataset_from_directory(
  directory="../augdata",
  labels="inferred",
  label_mode="binary",
  color_mode="rgb",
  batch_size=32,
  image_size=(224, 224),
  shuffle=True,
  validation_split=0.21,
  seed=32,
  subset="training",
)

val_gen = tf.keras.preprocessing.image_dataset_from_directory(
  directory="./data",
  labels="inferred",
  label_mode="binary",
  color_mode="rgb",
  batch_size=32,
  image_size=(224, 224),
  shuffle=True,
  seed=32,
  validation_split=0.21,
  subset="validation",
)

callback1 = endtraining()

normaliser = lambda x, y : (tf.cast(x, tf.float32)/255.0, y)

train_gen = train_gen.map(normaliser)
val_gen = val_gen.map(normaliser)

model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(256, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


bread = model.fit(train_gen, epochs=4, validation_data=val_gen, callbacks=[callback1])

plt.plot(bread.history['accuracy'], label="training accuracy")
plt.plot(bread.history['val_accuracy'], label="validation accuracy")
plt.xlabel('No. of epochs')
plt.ylabel('Accuracy')
plt.title('Model Training Summary')
plt.legend()
plt.show()