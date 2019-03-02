# Evolutionary_NN_for_HIV_octomer_binary_classification
*Work in progress*

## Project Venereal
This project is an attempt to build an evolutionary model using Python. The code is capable of building a neural network and choosing the optimal hyperparameters for a given dataset. These hyperparameters include the number of layers in the network, the number of neurons in the layer, the activation function, etc. The model is trained to perform binary classification of HIV octomers, with a -1 representing no cleavage and a +1 representing cleavage. Such a model could prove useful for th automatic selection of network hyperparameters, although it should be noted that reinforcement learning has proven to be more powerful than evolutionary algorithms. The applications for a model that predicts HIV cleavage could be instrumental in the search for early diagnosis methods and potential cures.

## Future plans and outcomes
I hope to build on my current model to encapsulate beyond the static hyperparameter limitations I outlined for the model. For example, instead of giving the program a handful of number-of-neurons-per-layer options to choose from, I hope to be able to let the program decide the exact number, unbounded by what is intended by the developer. Reinforcement learning would be a step in that direction.
