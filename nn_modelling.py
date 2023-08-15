from modelling import Modelling
from sklearn.metrics import mean_squared_error, r2_score
from tabular_data import clean_tabular_data, load_airbnb
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Any, Optional, Tuple
from torch.utils.tensorboard import SummaryWriter
import datetime, time
import itertools
import json
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

class AirbnbNightlyPriceRegressionDataset(Dataset):
    '''
    Loads the AirBNB data into a Dataset returning any items as a Torch tensor
    '''

    def __init__(self):
        super().__init__()

        # Load the tabular data from the CSV file
        self.df = pd.read_csv('listing.csv', delimiter=',')
        self.df = clean_tabular_data(self.df)
        self.X, self.y = load_airbnb(self.df, 'Price_Night')

    def __getitem__(self, idx):

        # Get a single sample from the dataset at the given index
        return (torch.tensor(self.X.iloc[idx]), torch.tensor(self.y.iloc[idx]))
    
    def __len__(self):

        # Return the total number of samples in the dataset
        return len(self.X)
    
class NN(nn.Module):
    '''
    Configuration of the Neural Network
    '''

    def __init__(self, input_size, output_size, config):
        super(NN, self).__init__()

        # Define the Neural Network architecture using config parameters
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(input_size, config['HIDDEN_LAYER_WIDTH']))
        self.fc_layers.extend([nn.Linear(config['HIDDEN_LAYER_WIDTH'], config['HIDDEN_LAYER_WIDTH']) for _ in range(config['MODEL_DEPTH'] - 1)])
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(config['HIDDEN_LAYER_WIDTH'], output_size)
        
    def forward(self, x):

        # Define the forward pass process
        for layer in self.fc_layers:
            x = layer(x)
            x = self.relu(x)
        x = self.fc_out(x)
        return x

class NeuralNetworkModelling(Modelling):
    '''
    Neural Network Modelling module trains and validates specific models based on specified hyperparameters to find the best metrics for that Neural Network.

    Models can be trained individually or part of list of different hyperparameters to determine the best hyperparameters to use.
    '''

    def __init__(self, labels='') -> None:
        # Base class required as the model is loaded in AirbnbNightlyPriceRegressionDataset for this modelling
        pass

    def train(self, model, train_loader, val_loader, test_loader, num_epochs, hyperparameters: dict):
        '''
        Trains a given model based on training and validation datasets passed in.

            Parameters:
                model - The model to be trained
                train_loader - The training dataset
                val_loader - The validation dataset
                num_epochs - Number of cycles to train the data
                hyperparameters: dict - A dictionary of the neural network's hyperparameters

            Returns:
                dict - Performance metrics of the trained model
        '''

        # Make a note of the time taken to train the model and keep a track of the average time to make a prediction
        start_time = time.time()
        inference_latency = 0

        # Move the model to the same device as the data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Define optimizer based on the configuration file
        if hyperparameters['OPTIMISER'] == 'SGD':
            optimiser = torch.optim.SGD(model.parameters(), lr=hyperparameters['LEARNING_RATE'])
        elif hyperparameters['OPTIMISER'] == 'ADAM':
            optimiser = torch.optim.Adam(model.parameters(), lr=hyperparameters['LEARNING_RATE'])
        else:
            raise ValueError(f"Unsupported optimizer: {hyperparameters['OPTIMISER']}")

        # Initialise a TensorBoard instance to plot the training curves
        writer = SummaryWriter()
        train_batch_idx = 0
        val_batch_idx = 0

        # Loop through each epoch
        for epoch in range(num_epochs):

            # Set the model to training mode
            model.train()

            # Store the training values
            train_predictions = []
            train_labels = []

            # Train the model
            train_predictions, train_labels, inference_latency, train_batch_idx, optimiser = self.get_prediction(
                train_loader, device, model, train_predictions, train_labels, inference_latency, writer, train_batch_idx, optimiser)
            
            # Set model to evaluation mode on validation set
            model.eval()

            # Store the validation values
            val_predictions = []
            val_labels = []

            # Validate the model
            with torch.no_grad():
                val_predictions, val_labels, inference_latency, val_batch_idx, _ = self.get_prediction(
                    val_loader, device, model, val_predictions, val_labels, inference_latency, writer, val_batch_idx)
                
            # Store the test values
            test_predictions = []
            test_labels = []

            # Test the model
            with torch.no_grad():
                test_predictions, test_labels, inference_latency, _, _ = self.get_prediction(
                    test_loader, device, model, test_predictions, test_labels, inference_latency)
                
            # Store the performance metrics found when training this model
            training_duration = time.time() - start_time
            num_inference = len(train_loader) + len(val_loader) + len(test_loader)
            average_inference_latency = inference_latency / num_inference

            # Store RMSE values
            train_rmse = mean_squared_error(train_labels, train_predictions, squared=False)
            val_rmse = mean_squared_error(val_labels, val_predictions, squared=False)
            test_rmse = mean_squared_error(test_labels, test_predictions, squared=False)

            # Store the R^2 values
            train_r2 = r2_score(train_labels, train_predictions)
            val_r2 = r2_score(val_labels, val_predictions)
            test_r2 = r2_score(test_labels, test_predictions)

            # Store the performance metrics in a dictionary
            performance_metrics = {
                'RMSE_loss': {
                    'train': float(train_rmse),
                    'val': float(val_rmse),
                    'test': float(test_rmse)
                },
                'R_squared': {
                    'train': float(train_r2),
                    'val': float(val_r2),
                    'test': float(test_r2)
                },
                'training_duration': round(training_duration, 4),
                'inference_latency': round(average_inference_latency, 6)
            }

        return performance_metrics
    
    def get_prediction(self, loader: DataLoader, device, model: NN, predictions_list: list, labels_list: list, inference_latency: float,
                       writer: SummaryWriter=None, batch_idx: int=None, optimiser=None) -> Tuple[list, list, float, Optional[int], Optional[Any]]:
        '''
        Produces a list of predictions and labels based on a passed in DataLoader.

        The method will also train the model when model is in training mode and plot the loss on a TensorBoard.

            Parameters:
                loader: DataLoader - The dataset of data to train, validate or test against
                device - The device the model is being run on
                model: NN - The model that is being trained
                predictions_list: list - The list of predictions that are currently stored against the model
                labels_list: list - The list of labels that are currently stored against the model
                inference_latency: float - The average time so far to determine a prediction
                writer: SummaryWriter (optional) - The instance of the TensorBoard that is being used to plot data
                batch_idx: int (optional) - The batch number that the TensorBoard is currently on to plot
                optimiser: (optional) - The instance of the optimiser to update if training the model

            Returns:
                list - Updated list of predictions
                list - Updated list of labels
                int - Updated inference latency after last prediction run
                Optional[int] - Latest batch number for the TensorBoard plot
                Optional[Any] - Updated optimiser after it has done a backward step
        '''
        # For each set of features and labels in the dataset
        for features, labels in loader:

            # Define the features and labels
            features = features.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.float32)

            # Get a prediction based on the model
            prediction_start = time.time()
            prediction = model(features)
            prediction_end = time.time()

            # Store the predication and labels for the metrics calculation later
            predictions_list.extend(prediction.cpu().detach().numpy())
            labels_list.extend(labels.cpu().detach().numpy())

            # Store the time it took to make a prediction overall so far
            inference_latency += prediction_end - prediction_start

            # If not the Test DataLoader
            if writer != None:
                # Reshape labels to match the shape of prediction
                labels = labels.view(-1, 1)

                # Calculate the MSE loss and do a backward pass if training the model
                loss = F.mse_loss(prediction, labels)
                if model.training == True:
                    loss.backward()

                # Plot the loss value in TensorBoard graph
                if model.training == True:
                    writer.add_scalar('train loss', loss.item(), batch_idx)
                else:
                    writer.add_scalar('val loss', loss.item(), batch_idx)
                batch_idx += 1

                # optimise the model
                if model.training == True:
                    optimiser.step()
                    optimiser.zero_grad()

        return predictions_list, labels_list, inference_latency, batch_idx, optimiser

    def get_nn_config(self, filename: str) -> dict:
            '''
            Gets the Neural Network architecture parameters from a specified yaml file

            Parameters:
                filename: str - The file path of the yaml file

            returns:
                dict - A dictionary of the Neural Network Architecture paramaters
            '''
            with open(filename, 'r') as file:
                config = yaml.safe_load(file)

                return config
    
    def generate_nn_configs(self, optimisers: list, learning_rates: list, hidden_layer_widths: list, model_depths: list) -> list:
        """
        Generates a list of configuration dictionaries with different combinations of hyperparameters
        
        Parameters:
            optimizers: list -  List of optimisers to try
            learning_rates: list - List of learning rates to try
            hidden_layer_widths: list - List of hidden layer widths to try
            model_depths: list - List of model depths (number of hidden layers) to try
            
        Returns:
            list - A list of configuration dictionaries
        """
        config_list = []
        
        # Loop through every combination of hyperparameters and store in a config list
        for optimiser, lr, width, depth in itertools.product(optimisers, learning_rates, hidden_layer_widths, model_depths):
            config = {
                'OPTIMISER': optimiser,
                'LEARNING_RATE': lr,
                'HIDDEN_LAYER_WIDTH': width,
                'MODEL_DEPTH': depth
            }
            config_list.append(config)
        
        return config_list

    def find_best_nn(self, train_loader, val_loader, test_loader, input_size, output_size, num_epochs):
        """
        Finds the best neural network model by training with different configurations
        
        Parameters:
            train_loader: DataLoader - Data loader for the training dataset
            val_loader: DataLoader - Data loader for the validation dataset
            test_loader: DataLoader - Data loader for the test dataset
            input_size: int - Size of the input features
            output_size: int - Size of the output
            num_epochs: int - Number of epochs for training

        Returns:
            nn.Module - The best trained neural network model
            dict - Hyperparameters of the best model
            dict - Performance metrics of the best model
        """
        # Define lists of hyperparameters to try
        optimisers = ['SGD', 'ADAM']
        learning_rates = [0.0001, 0.001]
        hidden_layer_widths = [4, 8]
        model_depths = [1, 2]
        
        # Generate configurations
        configs = self.generate_nn_configs(optimisers, learning_rates, hidden_layer_widths, model_depths)
        
        # Store the best model results
        best_model = None
        best_metrics = None
        best_hyperparameters = None
        best_val_rmse = float('inf')
        
        # To store all hyperparameters
        all_hyperparameters = []
        
        for i, config in enumerate(configs):
            print(f"Training model with Configuration {i + 1}:", config)
            
            # Create an instance of the Neural Network
            model = NN(input_size, output_size, config)
            
            # Train the model
            performance_metrics = self.train(model, train_loader, val_loader, test_loader, num_epochs=num_epochs, hyperparameters=config)
            
            # Get validation RMSE from the metrics
            val_rmse = performance_metrics['RMSE_loss']['val']
            
            # Save the best model if validation RMSE is lower
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_model = model
                best_metrics = performance_metrics
                best_hyperparameters = config
            
            # Store hyperparameters
            all_hyperparameters.append(config)
        
        # Get the current date and time
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Set the folder name where the files should be saved by a timestamp
        folder = os.path.join('neural_networks', 'regression', current_datetime)

        # Save the model, hyperparameters and performance metrics
        self.save_model(best_model, best_hyperparameters, best_metrics, folder)

        # Save all hyperparameters to a JSON file
        hyperparameters_path = os.path.join('neural_networks', 'regression', current_datetime, 'all_hyperparameters.json')
        with open(hyperparameters_path, 'w') as hyperparameters_file:
            for hyperparam in all_hyperparameters:
                json.dump(hyperparam, hyperparameters_file)
                hyperparameters_file.write('\n')
            
        return best_model, best_hyperparameters, best_metrics

if __name__ == "__main__":
    # Neural Network test code:
    
    # Create an instance of the AirBNB dataset 
    dataset = AirbnbNightlyPriceRegressionDataset()

    # Define the size of train, validation and testing sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # Split the dataset into train, validation and test sets
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Define batch size for data loaders
    batch_size = 16

    # Create data loaders for train, validation and test sets with shuffling
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Define the input size based on the training set size and output size for the model
    input_size = len(next(iter(train_loader))[0][0])
    output_size = 1

    # Create an instance of the Neural Network Modelling class
    nn_model = NeuralNetworkModelling()

    # Get the best Neural Network Model available
    best_model, best_hyperparameters, best_metrics = nn_model.find_best_nn(train_loader, val_loader, test_loader, input_size, output_size, num_epochs=50)

    # Print the results of the best model:
    print()
    print(f"Best Model Hyperparameters -", best_hyperparameters)
    print(f"Performance Metrics -", best_metrics)