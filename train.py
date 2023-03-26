# import keras.api._v2.keras as keras
import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt

train_dataset = keras.utils.image_dataset_from_directory('data/train/',
                                                         shuffle=True,
                                                         batch_size=32,
                                                         image_size=(224, 224))
valid_dataset = keras.utils.image_dataset_from_directory('data/val/',
                                                         shuffle=True,
                                                         batch_size=32,
                                                         image_size=(224, 224))
test_dataset = keras.utils.image_dataset_from_directory('data/test/',
                                                        shuffle=True,
                                                        batch_size=32,
                                                        image_size=(224, 224))
class_names = test_dataset.class_names

w,h,channels = 224,224,3

base_model = keras.applications.ResNet50(include_top=False, input_shape=(224,224,channels))

base_model.trainable = False
inputs = keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(0.25)(x)
x = keras.layers.Dense(512, activation='relu', use_bias=False)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(0.5)(x)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

model.summary()

# Parameters
learning_rate = 0.01
batch_size = 32
epochs=4

model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_dataset,
          batch_size=batch_size, epochs=1,
          verbose=1,
          validation_data=valid_dataset)
keras.backend.set_value(model.optimizer.learning_rate, model.optimizer.learning_rate.numpy()/2)
model.layers[0].trainable = True
model.fit(train_dataset,
          batch_size=batch_size, epochs=epochs,
          verbose=1,
          validation_data=valid_dataset)

# Retrieve a batch of images from the test set
image_batch, label_batch = test_dataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch).flatten()

# Apply a sigmoid since our model returns logits
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1)

print('Predictions:\n', predictions.numpy())
print('Labels:\n', label_batch)

plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].astype("uint8"))
  plt.title(class_names[predictions[i]])
  plt.axis("off")
plt.show()