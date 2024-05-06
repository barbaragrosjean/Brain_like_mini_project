# Predicting IT neuron activity
NX-414 Brain-like computation and intelligence, EPFL

The objective of this project is to investigate various modeling approaches for predicting neural activity originating from the inferior temporal (IT) cortex in monkeys when presented with visual stimuli during an object recognition task.

## Requirements
- Python >= 3.5
- Numpy
- PyTorch
- Matplotlib
- Sklearn
- optuna
- Pandas
- Torchvision

## Usage
The code is set for reproducibility with fixed random seeds for numpy, pytorch and cuda. As our models are heavy to train, we recommend to use a GPU. The code is set to run on a GPU if available, otherwise it will run on a CPU.

## Dataset
The datasets utilized are sourced from Majaj et
al. (Journal of Neuroscience, 2015), and have undergone preprocessing,
encompassing neural recordings obtained via multielectrode arrays
implanted in the IT cortex of monkeys (168 neurons), alongside images
of objects set against a natural landscape background (RGB channels,
224 × 224 pixels). 

## Notebooks
This project comprises four notebooks: week6.ipynb (Ridge linear regression from input pixels and task-driven approach), week7.ipynb (data-driven approach), week9.ipynb (model exploration), test.ipynb (evaluation of our best performing model). 
**Each notebook needs to be run in the same folder as the utils.py file.** Each notebook encompasses model training and evaluation. The metrics used are the mean explained variance and correlation along each neuron. The data can be downloaded directly from all the notebooks using the utils.py module. 

#### week6.ipynb
The second part of this notebook (task-driven approach) requires either extracting and storing (Section A) or directly loading the previously saved layer activations from the PyTorch ResNet50 (Section B). The second option requires the files *ACTIVATIONS* and *ACTIVATIONS_RND* to be in the same folder as week6.ipynb. 
The *functions.py* file also needs to be located in the same folder.

#### week7.ipynb
A data-driven approach is implemented in this notebook. The model can either be trained in a first place, but can also be directly loaded from the *c3l1.pth* file for further evaluation.

#### week9.ipynb
Three different models are trained independently in this notebook. The second model is our best performing model.

#### test.ipynb
This notebook comprises the evaluation of our best-performing model: a 3-convolutional-layer deep neural network followed by 2 fully-connected layers, trained on top of the ResNet50 third convolutional layer. The notebook needs to be run in the same folder as the model file *res_cnn_layer3.pth*.
