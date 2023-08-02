import pandas as pd

def remove_rows_with_missing_ratings(df: pd.DataFrame) -> pd.DataFrame:
    '''
    This method removes all rows in a DataFrame that contain NA values in the Ratings columns

    df: Pandas Dataframe - The Dataframe that is going to be cleaned

    returns: Pandas Dataframe - DataFrame with cleaned rows containing a rating value
    '''

    # Drop records with missing ratings
    df = df.dropna(subset=['Cleanliness_rating'])
    # In example all rows the same with missing ratings, but just in case check all ratings columns
    df = df.dropna(subset=['Accuracy_rating'])
    df = df.dropna(subset=['Location_rating'])
    df = df.dropna(subset=['Check-in_rating'])
    df = df.dropna(subset=['Value_rating'])

    return df

def combine_description_strings(df: pd.DataFrame) -> pd.DataFrame:
    '''
    This method removes all rows in a DataFrame that contain no description and converts the description
    into a single string rather than a list of strings

    Parameters:
        df: Pandas Dataframe - The Dataframe that contains the Description field that will be cleaned

    Returns:
        Pandas Dataframe - DataFrame with a cleaned up Description field
    '''

     # Drop records with missing descriptions
    df.dropna(subset=['Description'], inplace=True)

    # Clean the description to remove the "About this space" prefix and remove any whitespace
    df['Description'] = df['Description'].str.replace(r'About this space', '', regex=True).str.strip()
    df['Description'] = df['Description'].str.replace(r'^\s*\'\',*\s*', '', regex=True).str.strip()
    df['Description'] = df['Description'].str.replace(r'[\[\],\']', '', regex=True).str.strip()

    return df

def set_default_feature_values(df: pd.DataFrame) -> pd.DataFrame:
    '''
    This method replaces any empty values to 1 for the following columns:
    guests, beds, bathrooms, bedrooms

    Parameters:
        df: Pandas Dataframe - The Dataframe that will update the empty values to 1

    Returns:
        Pandas Dataframe - DataFrame with default values applied
    '''
    
    default_values = {'guests': 1, 'beds': 1, 'bathrooms': 1, 'bedrooms': 1}
    df.fillna(default_values, inplace=True)
    # Drop bad column from dataset as not required
    df.drop(columns=['Unnamed: 19'], inplace=True)

    return df

def clean_tabular_data(df: pd.DataFrame) -> pd.DataFrame:
    '''
    This method calls all the assocaited methods to clean the dataframe

    Parameters:
        df: Pandas Dataframe - The Dataframe that is to be cleaned

    Returns:
        Pandas Dataframe - DataFrame with cleaned up data
    '''

    df = remove_rows_with_missing_ratings(df)
    df = combine_description_strings(df)
    df = set_default_feature_values(df)

    return df

def load_airbnb(df: pd.DataFrame, label: str) -> tuple:
    '''
    This method gets the list of features and labels of the data

    Parameters:
        df: Pandas Dataframe - The Dataframe that contains the features and labels
        label: str - The name of the column that will be the label

    Returns:
        tuple - A tuple of features and labels
    '''

    features = df.drop(columns=[label])
    labels = df[label]

    # Filter out columns containing text data for the features
    features = features.select_dtypes(include='number')

    return features, labels

if __name__ == "__main__":
    airbnb_df = pd.read_csv('listing.csv', delimiter=',')
    airbnb_df = clean_tabular_data(airbnb_df)
    features, labels = load_airbnb(airbnb_df, 'Price_Night')
    print(features)
    print(labels)
    