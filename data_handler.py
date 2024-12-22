import pandas as pd

def load_csv():
    """
    Loads a CSV file and validates its structure.

    Returns: 
        pd.DataFrame: A DataFrame containing the data from the CSV file.
    """
    try:
        df = pd.read_csv('imdb_tvshows.csv')
    except FileNotFoundError:
        print("The file 'imdb_tvshows.csv' was not found. Please make sure it exists in the current directory.")
        exit(1)
    try:
        assert 'Title','Description' in df.columns
    except AssertionError:
        print("The CSV file must have columns named 'Title' and 'Description'.")
        exit(1)
    return df