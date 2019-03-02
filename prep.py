import logging
import numpy as np
from tqdm import tqdm
from functools import reduce
from operator import add
import random
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
import optim

"""DATA PREPROCESSING PT. 1"""
# Open dataset and append items into a new list
dataset = []

with open("schillingData.txt") as file:
    for line in file:
        line = str(line)
        dataset.append(line)

# Create a function to split by the "," delimeter and remove the "\n" newline character
def split_and_remove_newline(opened_file):

    cleaned_dataset = []

    for line in dataset:
        line = line.translate({ord(char): None for char in "\n"})
        newline = line.split(",")
        cleaned_dataset.append(newline)
    return cleaned_dataset

octomer_cleavage_pairs = split_and_remove_newline(dataset)

# Seperate the data pairs into 2 seperate datasets for data preprocessing
def separator(data_pairs):
    octomers = list([octomer[0] for octomer in data_pairs])
    cleavages = list([cleavage[-1] for cleavage in data_pairs])
    
    return octomers, cleavages

octomers_dataset, cleavages_dataset = separator(octomer_cleavage_pairs)

"""DATA PREPROCESSING PT.2"""
# Create unique character dictionary for octomers
charset = set("".join(list(octomers_dataset)))
char_to_int = dict((c, i) for i, c in enumerate(charset))
int_to_char = dict((i, c) for i, c in enumerate(charset))

charset_len = len(charset)
dataset_len = len(octomers_dataset)

print ("Number of training samples: %s" %dataset_len)
print ("Number of unique characters: %s" %charset_len)

# Integerize the octomers using the dictionary we created
X_data = []

def int_and_norm(octomers):
    for octomer in octomers:
        X_data.append([(float(char_to_int[char]) / float(charset_len)) for char in octomer])
    return X_data
  
int_and_norm(octomers_dataset)

X_data = [[float(thing) for thing in item] for item in X_data]

# Turn -1 to 0 in the cleavages dataset to make for easier training
Y_data = [0 if float(cleavage) == -1 else float(cleavage) for cleavage in cleavages_dataset]

assert len(X_data) == len(Y_data), "X and Y data are not the same length"

# One hot encode Y data 
Y_data = [[1, 0] if number == 1.0 else [0, 1] for number in Y_data]

# Split dataset into testing and training data
def data_splitter(dataset):
    split_len = int(dataset_len / 3 * 2)

    train_dataset = np.array(dataset[:split_len])
    test_dataset = np.array(dataset[split_len:])

    return train_dataset, test_dataset

"""EVOLUTIONARY NETWORK"""
early_stopper = EarlyStopping(patience = 5)

def dataloader():
    nb_classes = 2
    batch_size = 1
    input_shape = (8, )
    
    X_data_train, X_data_test = data_splitter(X_data)
    Y_data_train, Y_data_test = data_splitter(Y_data)

    return (nb_classes, batch_size, input_shape, X_data_train, X_data_test, Y_data_train, Y_data_test)

def compile_model(network, nb_classes, input_shape):
    """Compile a sequential model.
    Args:
        network (dict): the parameters of the network
    Returns:
        a compiled network.
    """
    # Get our network parameters.
    nb_layers = network["nb_layers"]
    nb_neurons = network["nb_neurons"]
    activation = network["activation"]
    optimizer = network["optimizer"]

    model = Sequential()

    # Add each layer.
    for i in range(nb_layers):

        # Need input shape for first layer.
        if i == 0:
            model.add(Dense(nb_neurons, activation = activation, input_shape = input_shape))
        else:
            model.add(Dense(nb_neurons, activation = activation))

        model.add(Dropout(0.2))  # hard-coded dropout

    # Output layer.
    model.add(Dense(nb_classes, activation = 'softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer = optimizer,
                  metrics = ['accuracy'])

    return model

def train_and_score(network, dataset):
    if dataset == "venereal":
        nb_classes, batch_size, input_shape, X_data_train, X_data_test, Y_data_train, Y_data_test = dataloader()
    else:
        print ("No dataset selected")

    model = compile_model(network, nb_classes, input_shape)

    model.fit(X_data_train, Y_data_train,
            batch_size = batch_size,
            epochs = 100000,  # using early stopping, so no real limit
            verbose = 0,
            validation_data = (X_data_test, Y_data_test),
            callbacks = [early_stopper])

    score = model.evaluate(X_data_test, Y_data_test, verbose = 0)

    return score[1]  # 1 is accuracy. 0 is loss.

"""RUNNING TASKS"""
# Setup logging.
logging.basicConfig(
    format = '%(asctime)s - %(levelname)s - %(message)s',
    datefmt = '%m/%d/%Y %I:%M:%S %p',
    level = logging.DEBUG,
    filename = 'log.txt'
)

def train_networks(networks, dataset):
    """Train each network.
    Args:
        networks (list): Current population of networks
        dataset (str): Dataset to use for training/evaluating
    """
    pbar = tqdm(total = len(networks))
    for network in networks:
        network.train(dataset)
        pbar.update(1)
    pbar.close()

def get_average_accuracy(networks):
    """Get the average accuracy for a group of networks.
    Args:
        networks (list): List of networks
    Returns:
        float: The average accuracy of a population of networks.
    """
    total_accuracy = 0
    for network in networks:
        total_accuracy += network.accuracy

    return total_accuracy / len(networks)

def generate(generations, population, nn_param_choices, dataset):
    """Generate a network with the genetic algorithm.
    Args:
        generations (int): Number of times to evole the population
        population (int): Number of networks in each generation
        nn_param_choices (dict): Parameter choices for networks
        dataset (str): Dataset to use for training/evaluating
    """
    optimizer = optim.Optimizer(nn_param_choices)
    networks = optimizer.create_population(population)

    # Evolve the generation.
    for i in range(generations):
        logging.info("***Doing generation %d of %d***" %
                     (i + 1, generations))

        # Train and get accuracy for networks.
        train_networks(networks, dataset)

        # Get the average accuracy for this generation.
        average_accuracy = get_average_accuracy(networks)

        # Print out the average accuracy each generation.
        logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
        logging.info('-' * 80)

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Do the evolution.
            networks = optimizer.evolve(networks)

    # Sort our final population.
    networks = sorted(networks, key = lambda x: x.accuracy, reverse = True)

    # Print out the top 5 networks.
    print_networks(networks[:5])

def print_networks(networks):
    """Print a list of networks.
    Args:
        networks (list): The population of networks
    """
    logging.info('-' * 80)
    for network in networks:
        network.print_network()

def main():
    """Evolve a network."""
    generations = 10  # Number of times to evole the population.
    population = 20  # Number of networks in each generation.
    dataset = 'venereal'

    nn_param_choices = {
        'nb_neurons': [64, 128, 256, 512, 768, 1024],
        'nb_layers': [1, 2, 3, 4],
        'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
        'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad',
                      'adadelta', 'adamax', 'nadam'],
    }

    logging.info("***Evolving %d generations with population %d***" %
                 (generations, population))

    generate(generations, population, nn_param_choices, dataset)

if __name__ == '__main__':
    main()