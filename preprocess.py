import glob
from PIL import Image
import numpy as np
import random

data_path = "./data"
w, h = 224, 224
channels = 3
label_names = ['chihuahua', 'cat']

def load(path, type, label_names):
    assert(type in ['test', 'train', 'val'])
    images = []
    labels = []
    path = f"data/{type}"
    for filename in glob.iglob(f"{path}/*/*"):
        label = filename[:filename.rfind('/')]
        label = label[label.rfind('/')+1:]
        img = Image.open(filename)
        img = img.resize((w, h))
        np_img = np.asarray(img)

        if np_img.ndim != 3:
            print("skip")
            continue
        images.append(np_img / 255)
        labels.append((np.array(label_names) == label) * 1)
    
    return {'images': np.array(images), 'labels': np.array(labels)}

training_data = load(data_path, type="train", label_names=label_names)
test_data = load(data_path, type="test", label_names=label_names)
val_data = load(data_path, type="val", label_names=label_names)

# shuffle
shuffled_idx = list(range(len(training_data['images'])))
random.shuffle(shuffled_idx)
shuffled_idx = np.array(shuffled_idx)
training_data['images'] = training_data['images'][shuffled_idx]
training_data['labels'] = training_data['labels'][shuffled_idx]

import pickle

def save_as_pickle(name, data):
    with open(name, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

save_as_pickle('train', training_data)
save_as_pickle('test', test_data)
save_as_pickle('val', val_data)

print(training_data['images'].shape)
print(test_data['images'].shape)
print(val_data['images'].shape)
print(training_data['labels'].shape)
print(test_data['labels'].shape)
print(val_data['labels'].shape)