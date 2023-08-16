from itertools import product
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, precision_score, r2_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from tabular_data import clean_tabular_data, load_airbnb
import joblib, json
import numpy as np
import os
import pandas as pd
import torch

class Modelling:
    '''
    Core module for modelling an AirBNB DataFrame loaded from listing.csv to determine the best model to make predictions against the data.

    Best models will be saved in a specified folder locally so can be reused in future.
    '''

    def __init__(self, label: str) -> None:

        # Load the dataset into a Pandas DataFrame, clean it and produce the features and label
        self.df = pd.read_csv('listing.csv', delimiter=',')
        self.df = clean_tabular_data(self.df)
        self.features, self.label = load_airbnb(self.df, label)

    def save_model(self, model, hyperparameters: dict, performance_metrics: dict, folder: str):
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
        if isinstance(model, torch.nn.Module):
            model_path = os.path.join(folder, 'model.pt')
            torch.save(model, model_path)
        else:
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

        print(f"The model has been saved to the following folder - {folder}")

    def find_best_model(self, root_folder: str):
        '''
        Evaluates all the model metrics stored within a specified folder structure 

        Parameters:
            root_folder: str - The file path of the folder containing the sub folders of model data

        Returns:
            best_model - The best regression model
            best_hyperparameters: dict - Dictionary of the hyperparameters for the best model
            best_metrics: dict - Dictionary of the performance metrics for the best model
        '''
        
        # Check if a Regression or Classification Model
        if "regression" in root_folder:
            is_regression = True
        elif "classification" in root_folder:
            is_regression = False
        else:
            raise Exception("Expected the folder name regression or classification in ", root_folder)

        # Store the best model outputs whilst searching through the files
        if is_regression == True:
            best_value = np.inf
        else:
            best_value = 0
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

                    # If a Regression model
                    if is_regression == True:

                        # Regression - Retrieve the Validation RMSE value from the JSON file 
                        value = metrics_data.get('validation_RMSE', None)
                    else:
                        # Classification - Retrieve the Validation Accuracy value from the JSON file 
                        value = metrics_data.get('validation_accuracy', None)

                    # If the RMSE value is lower than the best RMSE found so far
                    if self.is_best_score(is_regression, value, best_value):

                        # Store the best model's metrics data
                        best_value = value
                        best_metrics = metrics_data

                        # Store the best model
                        model_file = os.path.join(folder_path, 'model.joblib')
                        best_model = joblib.load(model_file)

                        # Store the best model's hyperparameters
                        hyperparameters_file = os.path.join(folder_path, 'hyperparameters.json')
                        with open(hyperparameters_file, 'r') as f2:
                            best_hyperparameters = json.load(f2)

        return best_model, best_hyperparameters, best_metrics
    
    def is_best_score(self, is_regression: bool, value: float, current_best_value: float) -> bool:
        '''
        Checks where a Regression or Classification performance value is better than the current best value

        Parameters:
            is_regression: bool - Regression == Ture, Classification == False
            value: float - Regression RMSE value or Classification Accuracy value
            current_best_value: float - Current best Regression RMSE value or Classification Accuracy value

        Returns:
            bool - True if value is a better score or False if not
        '''

        # If a Regression model
        if is_regression == True:

            # Regression - Check if RMSE value is less than the current best RMSE value
            if value < current_best_value:
                return True
            else:
                return False
        else:

            # Classification - Check if the Accuracy value is greater than the current best Accuracy value
            if value > current_best_value:
                return True
            else:
                return False
        

class RegressionModelling(Modelling):
    '''
    Regression Modelling module tunes specific Regression models based on specified hyperparameters to find the best metrics for that model.
    '''

    def __init__(self, label: str) -> None:
        super().__init__(label)

        self.regression_folder = f'models/regression/label_{label}/'

    def custom_tune_regression_model_hyperparameters(
            self, model_class, X_train, y_train, X_val, y_val, X_test, y_test, hyperparameters: dict):
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
            self, model_class, X_train, y_train,X_val, y_val,X_test, y_test, hyperparameters: dict):
        """
        Perform hyperparameter tuning using GridSearchCV

        Parameters:
            model_class - The regression model to be tuned
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
        print()
        print("Tuning model -", model_class.__class__.__name__, ":", hyperparameters)

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

    def evaluate_all_models(self):
        '''
        Evaluates all Regression models and saves the output to the models folder

        Parameters:
            None

        Returns:
            Nothing
        '''

        # Split the data into training, validation and testing sets
        X_train, X_temp, y_train, y_temp = train_test_split(self.features, self.label, test_size=0.3)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

        # Define a list of tuples, each containing the model class and its hyperparameters
        models = [
            (SGDRegressor(), {
                'alpha': [0.0001, 0.001, 0.01],
                'max_iter': [500, 1000, 2000],
                'learning_rate': ['constant', 'optimal', 'invscaling'],
                'penalty': ['l2', 'l1', 'elasticnet']
            }),
            (DecisionTreeRegressor(), {
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }),
            (RandomForestRegressor(), {
                'n_estimators': [50, 100],
                'max_depth': [None, 10],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2']
            }),
            (GradientBoostingRegressor(), {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2']
            })
        ]

        for model, hyperparameters in models:

            # Determine the best model and its metrics with the SKLearn's GridSerachCV method
            best_model, best_hyperparameters, performance_metrics = self.tune_regression_model_hyperparameters(
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
            folder = self.regression_folder + model.__class__.__name__

            # Print the best hyperparameters and performance metrics for that model
            print("Best Results for", best_model.__class__.__name__, ":")
            print(best_hyperparameters)
            print(performance_metrics)

            # Save the model, hyperparameters and performance metrics
            self.save_model(best_model, best_hyperparameters, performance_metrics, folder)

class ClassificationModelling(Modelling):
    '''
    Classification Modelling module tunes specific Classification models based on specified hyperparameters to find the best metrics for that model.
    '''

    def __init__(self, label: str) -> None:
        super().__init__(label)

        self.classification_folder = f'models/classification/label_{label}/'

    def tune_classification_model_hyperparameters(
            self, model_class, X_train, y_train, X_val, y_val, X_test, y_test, hyperparameters: dict):
        """
        Perform hyperparameter tuning using GridSearchCV

        Parameters:
            model - The classification model to be tuned
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
        print()
        print("Tuning model -", model_class.__class__.__name__, ":", hyperparameters)

        # Create GridSearchCV object with the provided model, hyperparamters and scoring metric
        grid_search = GridSearchCV(estimator=model_class, param_grid=hyperparameters, scoring='accuracy', cv=5, error_score='raise') 

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
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_precision = precision_score(y_val, y_val_pred, average='weighted', zero_division=1)
        val_recall = recall_score(y_val, y_val_pred, average='weighted', zero_division=1)
        val_f1_score = f1_score(y_val, y_val_pred, average='weighted', zero_division=1)

        # Fit the best model on the combined training and validation sets
        X_train_val_combined = np.vstack((X_train, X_val))
        y_train_val_combined = np.concatenate((y_train, y_val))
        best_model.fit(X_train_val_combined, y_train_val_combined)

        # Evaluate the best model on the test set
        y_test_pred = best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=1)
        test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=1)
        test_f1_score = f1_score(y_test, y_test_pred, average='weighted', zero_division=1)

        # Prepare the dictionary of performance metrics
        performance_metrics = {
            "validation_accuracy": val_accuracy,
            "validation_precision": val_precision,
            "validation_recall": val_recall,
            "validation_f1_score": val_f1_score,
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1_score": test_f1_score
        }

        return best_model, best_params, performance_metrics
    
    def evaluate_all_models(self):
        '''
        Evaluates all Classification models and saves the output to the models folder

        Parameters:
            None

        Returns:
            Nothing
        '''

        # Split the data into training, validation and testing sets
        X_train, X_temp, y_train, y_temp = train_test_split(self.features, self.label, test_size=0.3)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

        # Define a list of tuples, each containing the model class and its hyperparameters
        models = [
            (SGDClassifier(), {
                'alpha': [0.000001, 0.00001, 0.0001],
                'max_iter': [2000, 5000, 10000],
                'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
                'eta0': [0.001, 0.01],
                'penalty': ['l2', 'l1', 'elasticnet']
            }),
            (DecisionTreeClassifier(), {
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }),
            (RandomForestClassifier(), {
                'n_estimators': [50, 100],
                'max_depth': [None, 10],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2']
            }),
            (GradientBoostingClassifier(), {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.05],
                'max_depth': [3, 5],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2']
            })
        ]

        for model, hyperparameters in models:

            # Determine the best model and its metrics with the SKLearn's GridSerachCV method
            best_model, best_hyperparameters, performance_metrics = self.tune_classification_model_hyperparameters(
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
            folder = self.classification_folder + model.__class__.__name__

            # Print the best hyperparameters and performance metrics for that model
            print("Best Results for", best_model.__class__.__name__, ":")
            print(best_hyperparameters)
            print(performance_metrics)

            # Save the model, hyperparameters and performance metrics
            self.save_model(best_model, best_hyperparameters, performance_metrics, folder)

if __name__ == "__main__":

    # Regression Modelling test code:

    # Train the regression models for labels - Price_Night & bedrooms
    labels = ['Price_Night', 'bedrooms']
    for label in labels:
        print(f"*** Finding best model for label - {label} ***")

        regression_model = RegressionModelling(label=label)
        regression_model.evaluate_all_models()
        model, hyperparameters, performance_metrics = regression_model.find_best_model(root_folder=regression_model.regression_folder)

        print()
        print(f"*** Best Overall Regression Model (label = {label}) ***")
        print(model.__class__.__name__)
        print(hyperparameters)
        print(performance_metrics)
        print()
    
    # Classification Modelling test code:
    print(f"*** Finding best model for label - Category ***")

    classification_model = ClassificationModelling(label='Category')
    classification_model.evaluate_all_models()
    model, hyperparameters, performance_metrics = classification_model.find_best_model(root_folder=classification_model.classification_folder)

    print()
    print("*** Best Overall Classification Model (label = Category)  ***")
    print(model.__class__.__name__)
    print(hyperparameters)
    print(performance_metrics)
    print()