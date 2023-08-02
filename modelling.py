import joblib, json
import numpy as np
import os
import pandas as pd
from itertools import product
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor
from tabular_data import clean_tabular_data, load_airbnb

def custom_tune_regression_model_hyperparameters(
        model_class, X_train, y_train, X_val, y_val, X_test, y_test, hyperparameters: dict):
    """
    Performs a grid search over a reasonable range of hyperparameter values

    Parameters:
        model - The regression model to be tuned
        X_train - Training features
        y_train - Training labels
        X_val - Validation features
        y_val - Validation labels
        X_test - Testing features
        y_test - Testing labels
        hyperparameters: dict - Dictionary of the hyperparameters to use

    Returns:
        best_model - Best tuned model
        best_params: dict - Best hyperparameters found during the grid search
        performance_metrics: dict - Dictionary containing performance metrics on the validation set
    """
    
    # These are the results that need determining
    best_model = None
    best_hyperparameters = {}
    best_validation_rmse = np.inf

    # Perform grid search over hyperparameter values
    for params in product(*hyperparameters.values()):
        hyperparameter_values = dict(zip(hyperparameters.keys(), params))

        # Initialize the model with the hyperparameter values
        model = model_class(**hyperparameter_values)

        # Train the model on the training set
        model.fit(X_train, y_train)

        # Make predictions on the validation set
        y_val_pred = model.predict(X_val)

        # Calculate the RMSE on the validation set
        validation_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

        # Update best model and hyperparameters if current model performs better
        if validation_rmse < best_validation_rmse:
            best_model = model
            best_hyperparameters = hyperparameter_values
            best_validation_rmse = validation_rmse

    # Train the best model on the combined training and validation sets for best results
    X_train_val = np.vstack((X_train, X_val))
    y_train_val = np.concatenate((y_train, y_val))
    best_model.fit(X_train_val, y_train_val)

    # Evaluate the best model on the test set
    y_test_pred = best_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # Prepare the dictionary of performance metrics
    performance_metrics = {
        "validation_RMSE": best_validation_rmse,
        "test_RMSE": test_rmse
    }

    return best_model, best_hyperparameters, performance_metrics

def tune_regression_model_hyperparameters(
        model_class, X_train, y_train, X_val, y_val, X_test, y_test, hyperparameters: dict):
    """
    Perform hyperparameter tuning using GridSearchCV

    Parameters:
        model - The regression model to be tuned
        X_train - Training features
        y_train - Training labels
        X_val - Validation features
        y_val - Validation labels
        X_test - Testing features
        y_test - Testing labels
        hyperparameters: dict - Dictionary of the hyperparameters to use

    Returns:
        best_model - Best tuned model
        best_params - Best hyperparameters found during the grid search
        performance_metrics: dict - Dictionary containing performance metrics on the validation set
    """

    # Create GridSearchCV object with the provided model, hyperparamters and scoring metric
    grid_search = GridSearchCV(estimator=model_class, param_grid=hyperparameters, scoring='neg_mean_squared_error', cv=5, error_score='raise') 

    # Convert the data to NumPy arrays (removing feature names for warning message)
    X_train = X_train.values
    X_val = X_val.values
    X_test = X_test.values

    # Fit the GridSearchCV on the training data
    grid_search.fit(X_train, y_train)

    # Get the best model and its hyperparameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Evaluate the best model on the validation set
    y_val_pred = best_model.predict(X_val)

    # Calculate performance metrics on the validation set
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_r2 = r2_score(y_val, y_val_pred)

    # Fit the best model on the combined training and validation sets
    X_train_val_combined = np.vstack((X_train, X_val))
    y_train_val_combined = np.concatenate((y_train, y_val))
    best_model.fit(X_train_val_combined, y_train_val_combined)

    # Evaluate the best model on the test set
    y_test_pred = best_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)

    # Store the performance metrics in a dictionary
    performance_metrics = {
        'validation_RMSE': val_rmse,
        'validation_R^2': val_r2,
        'test_RMSE': test_rmse,
        'test_R^2': test_r2
    }

    return best_model, best_params, performance_metrics

def save_model(model, hyperparameters: dict, performance_metrics: dict, folder: str):
    '''
    Save the model information to a specified directory

    Parameters:
        model - The regression model
        hyperparameters: dict - Dictionary of the hyperparameters
        performance_metrics: dict - Dictionary of the performance metrics
        folder: str - The folder location to save the files

    Returns:
        Nothing
    '''
    
    # Create the folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # Save the model
    model_path = os.path.join(folder, 'model.joblib')
    joblib.dump(model, model_path)

    # Save the hyperparameters
    hyperparameters_path = os.path.join(folder, 'hyperparameters.json')
    with open(hyperparameters_path, 'w') as f:
        json.dump(hyperparameters, f)

    # Save the performance metrics
    metrics_path = os.path.join(folder, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(performance_metrics, f)

def evaluate_all_models():
    '''
    Evaluates all Regression models and saves the output to the models folder

    Parameters:
        None

    Returns:
        Nothing
    '''

    # Load the dataset into a Pandas DataFrame, clean it and produce the features and labels
    airbnb_df = pd.read_csv('listing.csv', delimiter=',')
    airbnb_df = clean_tabular_data(airbnb_df)
    features, labels = load_airbnb(airbnb_df, 'Price_Night')

    # Split the data into training, validation and testing sets
    X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

    # Define a list of tuples, each containing the model class and its hyperparameters
    models = [
        (SGDRegressor(), {
            "alpha": [0.0001, 0.001, 0.01],
            "max_iter": [500, 1000, 2000],
            "learning_rate": ["constant", "optimal", "invscaling"],
            "penalty": ['l2', 'l1', 'elasticnet']
        }),
        (DecisionTreeRegressor(), {
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }),
        (RandomForestRegressor(), {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }),
        (GradientBoostingRegressor(), {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        })
    ]

    for model, hyperparameters in models:

        # Determine the best model and its metrics with the SKLearn's GridSerachCV method
        best_model, best_hyperparameters, performance_metrics = tune_regression_model_hyperparameters(
            model_class=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            hyperparameters=hyperparameters
        )

        # Set the folder name where the files should be saved
        folder = 'models/regression/' + model.__class__.__name__

        # Save the model, hyperparameters, and performance metrics
        save_model(best_model, best_hyperparameters, performance_metrics, folder)

def find_best_model(root_folder: str):
    '''
    Evaluates all the model metrics stored within a specified folder structure 

    Parameters:
        root_folder: str - The file path of the folder containing the sub folders of model data

    Returns:
        best_model - The best regression model
        best_hyperparameters: dict - Dictionary of the hyperparameters for the best model
        best_metrics: dict - Dictionary of the performance metrics for the best model
    '''

    # Store the best model outputs whilst searching through the files
    best_rmse = np.inf
    best_model = None
    best_hyperparameters = {}
    best_metrics = {}

    # Loop through all folders in the root folder
    for folder in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder)
        if os.path.isdir(folder_path):

            # Retrive the metrics file and load it
            metrics_file = os.path.join(folder_path, 'metrics.json')
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)

                # Retrieve the Test RMSE value from the JSON file 
                test_rmse = metrics_data.get('test_RMSE', None)

                # If the RMSE value is lower than the best RMSE found so far
                if test_rmse < best_rmse:

                    # Store the best model's metrics data
                    best_rmse = test_rmse
                    best_metrics = metrics_data

                    # Store the best model
                    model_file = os.path.join(folder_path, 'model.joblib')
                    best_model = joblib.load(model_file)

                    # Store the best model's hyperparameters
                    hyperparameters_file = os.path.join(folder_path, 'hyperparameters.json')
                    with open(hyperparameters_file, 'r') as f2:
                        best_hyperparameters = json.load(f2)

    return best_model, best_hyperparameters, best_metrics


if __name__ == "__main__":
    evaluate_all_models()
    model, hyperparameters, performance_metrics = find_best_model(root_folder='models/regression')

    print("Best Model:")
    print(model.__class__.__name__)
    print(hyperparameters)
    print(performance_metrics)