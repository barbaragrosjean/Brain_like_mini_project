# Predicting IT neuron activity
NX-414 Brain-like computation and intelligence, EPFL

The objective of this project is to investigate various modeling approaches for predicting neural activity originating from the inferior temporal (IT) cortex in monkeys when presented with visual stimuli during an object recognition task.

## Requirements
- Python >= 3.5
- numpy
- pytorch
- matplotlib
- sklearn
- optuna
- pandas
- torchvision

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
This project contains four notebooks: week6.ipynb (Ridge linear regression from input pixels and task-driven approach), week7.ipynb (data-driven approach), week9.ipynb (model exploration), **test.ipynb** (best performing model). 
**Each notebook needs to be run in the same folder as the utils.py file.**

#### Load the data
The data can be downloaded directly from all the notebooks using the utils.py module. 

#### Week 6 Notebook

The Week6 notebook aims to develop the task-driven approach. The first part is about the methods to regress neural data to image stimuli. The second part is about using a pre-trained Resnet and a randomly initialized Resnet feed with the neural data to extract the layer activations at different levels and regress them into image stimuli. 
The models used are from the module torchvision.models that is installed by running the notebook.

Provided in the notebook: a way to store the figures and store and load the activations. 
Note that if you already have the activation stored in the folders Activations and Activations_RND, we can jump the part A. Extract the layer activations and go to section B. to load the saved activations.

#### Week 7 Notebook
The Week7 notebook aims to develop the data-driven approach. A shallow convolutional neural network is trained to fit the neural data and the stimuli. Previously to the train, a grid search is done for the training hyperparameter using the Optuna module. The model is store in the same repository as the notebook at the end.

#### Test Notebook
The test notebook aims to test the best-performing model from the exploration. The model chosen is a convolutional neural network train above the activations extracted from layer 3 of a pre-trained Resnet, combining the data-driven and task-driven approaches. It contains a way of loading the model: 'res_cnn_layer3.pth' and evaluating it. 
The notebook needs to be run in the same repository as the model.

#### Èvaluate the model
The metrics used are from sklearn.metrics library for the explained variance score and numpy library for the correlation. All the plots are done using matplotlib.pyplot library. 

