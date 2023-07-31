# Modelling Airbnb's property listing dataset

 This solution will build a framework to systematically train, tune, and evaluate models on several tasks that are tackled by the Airbnb team.

 ## Project Milestones for the Project:

### Milestone 1 - Data Preparation

> Before building the framework, the first goal was to understand how the Airbnb dataset is structured and clean it accordingly.

>> The source of the data is stored in listing.csv and the code to cleanse the data to get it ready to be analysed is in tabular_data.py

- Task 1: Loaded in the tabular dataset and functions written that cleanse the ratings, description and guest data fields.

- Task 2: Created a Pandas dataframe consisting of features that contain numerical data only and set the label as the "Price_night" as this will be what is determined in the model that will be created.