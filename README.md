# Distracted Drivers Starter Project

This starter project currently ranks in the top 15% of submissions and with a few minor changes can reach the top 10%.

## Training

Before using this code you must agree to the [Kaggle competition's terms of use](https://www.kaggle.com/c/state-farm-distracted-driver-detection).

### Setup

1. [Fork this repo](https://help.github.com/articles/fork-a-repo/).
2. [Clone the forked repo](https://help.github.com/articles/cloning-a-repository/).

### Dataset Preparation

1. Download the images and drivers list into the "dataset" folder:
2. Unzip both into the "dataset" folder so it looks like this:

  - distracted-drivers-tf/
    - dataset/
      - imgs/
        - train/
        - test/
      - driver_imgs_list.csv

3. From inside the dataset folder, run `python prep_dataset.py`. This generates a pickled dataset of numpy arrays.

### Running

This project uses [TensorFlow](https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html#download-and-setup).

To run the code locally simply install the dependencies and run `python main.py`.

## Model Development

1. Tune your learning rate. If the loss diverges, the learning rate is probably too high. If it never learns, it might be too low.
2. Good initialization is important. The initial values of weights can have a significant impact on learning. In general, you want the weights initialized based on the input and/or output dimensions to the layer (see Glorot or He initialization).
3. Early stopping can help prevent overfitting but good regularization is also beneficial. L1 and L2 can be hard to tune but batch normalization and dropout are usually much easier to work with.
