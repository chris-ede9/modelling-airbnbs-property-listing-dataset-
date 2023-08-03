# Modelling Airbnb's property listing dataset

 This solution will build a framework to systematically train, tune, and evaluate models on several tasks that are tackled by the Airbnb team.

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