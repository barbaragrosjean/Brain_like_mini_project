# Brain Like computation mini-project: Predicting IT neuron activity
NX-414 Brain-like computation and intelligence, EPFL

This project aims to explore different approaches and models to predict neural responses from the inferior temporal (IT) cortex and visual stimulus shown during an 
object recognition task on monkeys.

## Dataset
The data used are from the paper of Majaj, Hong et al. Weighted Sums of IT Firing Rates Predict Recognition Performance. It consists of recording from multielectrode arrays implanted in the IT cortex of monkeys (168 neurons),
alongside images of objects set against a natural landscape background (RGB channels, $24\times224$ pixels).}

## Requirment
- Python >= 3.5
- numpy
- pytorch
- matplotlib
- sklearn
- optuna
- pandas
- torchvision
- down h5py

## Example of use
This project contains three notebooks. 

#### Load the data
All the notebooks contain at the beginning, cells to load the data from a Google Drive repository. 
By running it you will be bring to install the gdown h5py module and use the utils file provided for the load and visualization functions. 
All the notebooks need to be run in the same repository as the utils.py files.

#### Formate the data
The data are packed into Dataset and Dataloader instances (torch.utils.data module).  

#### Week 6 Notebook

The Week6 notebook aims to develop the task-driven approach. The first part is about the methods to regress neural data to image stimuli. The second part is about using a pre-trained Resnet and a randomly initialized Resnet feed with the neural data to extract the layer activations at different levels and regress them into image stimuli. 
The models used are from the module torchvision.models that is installed by running the notebook.

Provided in the notebook: a way to store the figures and store and load the activations. 
Note that if you already have the activation stored in the folders Activations and Activations_RND, we can jump the part A. Extract the layer activations and go to section B. to load the saved activations.

#### Week 7 Notebook
The Week7 notebook aims to develop the data-driven approach. A shallow convolutional neural network is trained to fit the neural data and the stimuli. Previously to the train, a grid search is done for the training hyperparameter using the Optuna module. 

Provided in the notebook: a way to store and load the trained model. 

#### Test Notebook
The test notebook aims to test the best-performing model from the exploration. The model chosen is a convolutional neural network train above the activations extracted from layer 3 of a pre-trained Resnet, combining the data-driven and task-driven approaches. It contains a way of loading the model: 'res_cnn_layer3.pth' and evaluating it. 
The notebook needs to be run in the same repository as the model.

#### Èvaluate the model
The metrics used are from sklearn.metrics library for the explained variance score and numpy library for the correlation. All the plots are done using matplotlib.pyplot library. 

