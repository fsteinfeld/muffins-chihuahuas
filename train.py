import keras.api._v2.keras as keras
from keras import datasets, layers, models

# model = keras.applications.EfficientNetV2B0(weights=None,classes=2)
model = keras.applications.ResNet50(weights=None, classes=2)

