# Modelling Airbnb's property listing dataset

 This solution will build a framework to systematically train, tune and evaluate models on several tasks that are tackled by the Airbnb team.

## Application Setup:

> Please refer to the packages_list.txt file for required packages that need to be installed in the environment to run this application in a Python related IDE.

>  To train Regression and Classification models please refer to the modelling.py file. All hyperparameters are hardcoded within that file and can be altered to suit the model being trained. The models that have been trained can be found within the 'models' folder that is generated in the root directory.

> To tune a Neural Network please refer to the nn_modelling.py file. The hyperparameters can be configured within the nn_config.yaml file and used to find the best model available. The models that have been trained can be found within the 'neural_networks' folder that is generated in the root directory.

> TensorBoard is used in the Neural Network tuning to visualise the training loss and validation loss to help determine if the model used is a good fit.

## Project Milestones for the Project:

### Milestone 1 - Data Preparation

> Before building the framework, the first goal was to understand how the Airbnb dataset is structured and clean it accordingly.

>> The source of the data is stored in listing.csv and the code to cleanse the data to get it ready to be analysed is in tabular_data.py

Tasks for this milestone:

1. Loaded in the tabular dataset and functions written that cleanse the ratings, description and guest data fields.

2. Created a Pandas dataframe consisting of features that contain numerical data only and passed in a parameter for the label as this will be what is determined in the model that will be created. For the first example 'Price_Night' was used.

### Milestone 2 - Create a regression model

> The goal for this milestone was to create some machine learning models to predict the price of the listing per night and evaluate them. This was further extended to allow for any regression related label to be predicted, i.e Bedrooms.

>> The source file is modelling.py, class RegressionModelling

Tasks for this milestone:

1. Created a simple regression model to predict the nightly cost of each listing.

2. Evaluated the regression model preformance by computing the key measures, RMSE and R^2 for both the training and test sets.

3. Implemented a custom function to tune the hyperparameters of the model.

4. Created another function to tune the hyperparameters of the model using methods form sklearn, i.e GridSearchCV.

5. Created a function that would save a model to a specified folder, along with the hyperparameters and metrics in json files. Folder path - 'models/regression/label_{label name}'

6. Ran different models to improve performance of the chosen regression model. These models included Decision Trees, Random Forests and Gradient Boosting.

7. After running all the models the best overall model was found - RandomForestRegressor. Please note that this was only based on a limited amount of hyperparameter combinations being used and other models like GradientBoostingRegressor had an RMSE which was of a similar range.

```
*** Best Overall Regression Model (label = Price_Night) ***
RandomForestRegressor
{'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 50}
{'validation_RMSE': 89.20966116655171, 'validation_R^2': 0.41696482939935453, 'test_RMSE': 84.47028776826936, 'test_R^2': 0.24439508208370275}
```

### Milestone 3 - Create a Classification model

> The goal for this milestone was to create some machine learning models to predict the Category of the listing based on the numerical features.

>> The source file is modelling.py, class ClassificationModelling

Tasks for this milestone:

1. Created a simple classification model to predict the category of each listing.

2. Evaluated the classification model preformance by computing the key measures, Accurarcy, the F1 score, the precision and the recall for both the training and test sets.

3. Created a function to tune the hyperparameters of the model using methods form sklearn, i.e GridSearchCV.

4. Created a function that would save a model to a specified folder, along with the hyperparameters and metrics in json files. Folder path - 'models/classification/label_{label name}'

5. Ran different models to improve performance of the chosen regression model. These models included Decision Trees, Random Forests and Gradient Boosting.

6. After running all the models the best overall model was found - RandomForestClassifier. However the Accuracy score is low on all models and further analysis is required to see if there is a better model available based on the hyperparameters.

```
*** Best Overall Classification Model (label = Category)  ***
RandomForestClassifier
{'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 50}
{'validation_accuracy': 0.43548387096774194, 'validation_precision': 0.43934126436498355, 'validation_recall': 0.43548387096774194, 'validation_f1_score': 0.4189046734069725, 'test_accuracy': 0.328, 'test_precision': 0.3229034940291906, 'test_recall': 0.328, 'test_f1_score': 0.32062825983878623}
```

### Milestone 4 - Create a configurable Neural Network

> The goal for this milestone was to create a neural network to predict the nightly listing price from the numerical data in the tabular dataset.

>> The source file is nn_modelling.py, along with a yaml file for configuring the hyperparameters for finding the best model for the neural network is located in nn_config.yaml

Tasks for this milestone:

1. Created a PyTorch Dataset and DataLoader. The Dataset was created as AirbnbNightlyRegressionDataset and returns a tuple of (features, label) when indexed.The features are a tensor and the label is a scaler of the price per night.

2. Defined the PyTorch model class containing the architecture of the fully connected neural network.

3. Completed the training loop and training of the model to completion in the train method.

4. Used TensorBoard to visualise the training curves of the model and the accuracy both on the training and validation sets.

5. Created a configuration file to change the characteristics of the model based in a yaml file which can be used to modify the hyperparameters for all

6. Created a function that would save the model to a specified folder, along with the hyperparameters and metrics in json files.

7. Tuned the model, by creating a method that will pass in multiple variations of the hyperparameters to find the best solution. For the latest run, these were the results:

```
Best Model Hyperparameters - {'OPTIMISER': 'ADAM', 'LEARNING_RATE': 0.001, 'HIDDEN_LAYER_WIDTH': 8, 'MODEL_DEPTH': 2}
    
Performance Metrics - {'RMSE_loss': {'train': 110.359130859375, 'val': 130.18190002441406, 'test': 128.8540802001953}, 'R_squared': {'train': 0.17475878193826178, 'val': 0.18538126720813974, 'test': 0.21151047421909497}, 'training_duration': 11.7362, 'inference_latency': 0.008132}
```

Please note that this was only on a limited amount of hyperparameters and can be further explored by altering the hyperparameters in the nn_config.yaml file.