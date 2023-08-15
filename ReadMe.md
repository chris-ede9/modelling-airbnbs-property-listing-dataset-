# Modelling Airbnb's property listing dataset

 This solution will build a framework to systematically train, tune and evaluate models on several tasks that are tackled by the Airbnb team.

## Application Setup:

> Please refer to the packages_list.txt file for required packages that need to be installed in the environment to run this application in a Python related IDE.

 ## Project Milestones for the Project:

### Milestone 1 - Data Preparation

> Before building the framework, the first goal was to understand how the Airbnb dataset is structured and clean it accordingly.

>> The source of the data is stored in listing.csv and the code to cleanse the data to get it ready to be analysed is in tabular_data.py

- Task 1: Loaded in the tabular dataset and functions written that cleanse the ratings, description and guest data fields.

- Task 2: Created a Pandas dataframe consisting of features that contain numerical data only and set the label as the "Price_night" as this will be what is determined in the model that will be created.

### Milestone 2 - Create a regression model

> The goal for this milestone was to create some machine learning models to predict the price of the listing per night and evaluate them.

>> The source file is modelling.py, class RegressionModelling

- Task 1: Created a simple regression model to predict the nightly cost of each listing.

- Task 2: Evaluated the regression model preformance by computing the key measures, RMSE and R^2 for both the training and test sets.

- Task 3: Implemented a custom function to tune the hyperparameters of the model.

- Task 4: Created another function to tune the hyperparameters of the model using methods form sklearn, i.e GridSearchCV.

- Task 5: Created a function that would save a model to a specified folder, along with the hyperparameters and metrics in json files.

- Task 6: Ran different models to improve performance of the chosen regression model. These models included Decision Trees, Random Forests and Gradient Boosting.

- Task 7: After running all the models the best overall model was found - RandomForestRegressor. Please note that this was only based on a limited amount of hyperparameter combinations being used and other models like GradientBoostingRegressor had an RMSE which was of a similar range.

### Milestone 3 - Create a Classification model

> The goal for this milestone was to create some machine learning models to predict the Category of the listing based on the numerical features.

>> The source file is modelling.py, class ClassificationModelling

- Task 1: Created a simple classification model to predict the category of each listing.

- Task 2: Evaluated the classification model preformance by computing the key measures, Accurarcy, the F1 score, the precision and the recall for both the training and test sets.

- Task 3: Created a function to tune the hyperparameters of the model using methods form sklearn, i.e GridSearchCV.

- Task 4: Created a function that would save a model to a specified folder, along with the hyperparameters and metrics in json files.

- Task 5: Ran different models to improve performance of the chosen regression model. These models included Decision Trees, Random Forests and Gradient Boosting.

- Task 6: After running all the models the best overall model was found - RandomForestClassifier. However the Accuracy score is low on all models and further analysis is required to see if there is a better model available based on the hyperparameters.

### Milestone 4 - Create a configurable Neural Network

> The goal for this milestone was to create a neural network to predict the nightly listing price from the numerical data in the tabular dataset.

>> The source file is nn_modelling.py, along with a yaml file for configuring a single neural network located in nn_config.yaml

- Task 1: Created a PyTorch Dataset and DataLoader. The Dataset was created as AirbnbNightlyRegressionDataset and returns a tuple of (features, label) when indexed.The features are a tensor and the label is a scaler of the price per night.

- Task 2: Defined the PyTorch model class containing the architecture of the fully connected neural network.

- Task 3: Completed the training loop and training of the model to completion in the train method.

- Task 4: Used TensorBoard to visualise the training curves of the model and the accuracy both on the training and validation sets.

- Task 5: Created a configuration file to change the characteristics of the model based in a yaml file which can be used to modify the hyperparameters.

- Task 6: Created a function that would save the model to a specified folder, along with the hyperparameters and metrics in json files.

- Task 7: Tuned the model, by creating a method that will pass in multiple variations of the hyperparameters to find the best solution. For the latest run, these were the results:

    Best Model Hyperparameters - {'OPTIMISER': 'ADAM', 'LEARNING_RATE': 0.001, 'HIDDEN_LAYER_WIDTH': 8, 'MODEL_DEPTH': 2}
    
    Performance Metrics - {'RMSE_loss': {'train': 110.359130859375, 'val': 130.18190002441406, 'test': 128.8540802001953}, 'R_squared': {'train': 0.17475878193826178, 'val': 0.18538126720813974, 'test': 0.21151047421909497}, 'training_duration': 11.7362, 'inference_latency': 0.008132}

    Please note that this was only on a limited amount of hyperparameters.