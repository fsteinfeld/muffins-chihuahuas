import numpy as np
import random
import matplotlib.pyplot as plt
import pickle


def load_data(name):
    with open(f'{name}', 'rb') as handle:
        return pickle.load(handle)

training_data = load_data('train')

def show_sample_images(dataset):
    # these two variables are "the parameters" of this cell
    w = 10
    h = 10

    # this function uses the open, resize and array functions we have seen before
    _, axes_list = plt.subplots(h, w, figsize=(2*w, 2*h)) # define a grid of (w, h)

    for axes in axes_list:
        for ax in axes:
            ax.axis('off')
            idx = random.randrange(0, len(dataset['images']))
            ax.imshow(dataset['images'][idx]) # load and show
            # ax.set_title(dataset['labels'][idx])
    # plt.show()
    plt.savefig('figure.pdf')
            
show_sample_images(training_data)