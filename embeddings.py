import os
import pickle
from AI_models import connect_to_openai
from data_handler import load_csv


def pickle_hit_or_miss():
    """
    Checks for the existence of a pickle file and handles the loading or creation of TV show embeddings.

    Returns:
        dict: A dictionary of TV show embeddings, where keys are show names and values are their corresponding vectors (show name: vector).
    """
    if os.path.exists("tv_shows_embeddings.pkl"):
        return load_embedding_from_pickle()
    else:
        return save_new_embedding_to_pickle()
    

def load_embedding_from_pickle():
    """
    loads the embeddings from the pickle file
    Deserialize the dictionary (show name: vector) and return it

    Returns:
        dict: A dictionary of TV show embeddings, where keys are show names and values are their corresponding vectors.
    """
    try:
        with open("tv_shows_embeddings.pkl", "rb") as pickle_file:
            deserialized_dict = pickle.load(pickle_file)
            return deserialized_dict
    except Exception as e:
        print(f"An error occurred while loading the embeddings from the pickle file: {e}")


def save_new_embedding_to_pickle():
    """
    save the generate_embeddings to a pickle file such that we will call the embeddings API of open ai once on each of the shows’ descriptions

    Returns:
        dict: A dictionary containing the generated embeddings (show name: vector).
    """
    df = load_csv()
    dict_shows_vectors = generate_embeddings(df['Title'].tolist(), df['Description'].tolist())
    # serialize the dictionary to a pickle file
    try:
        with open("tv_shows_embeddings.pkl", "wb") as pickle_file:
            pickle.dump(dict_shows_vectors, pickle_file)
    except Exception as e:
        print(f"An error occurred while saving the embeddings to a pickle file: {e}")


def generate_embeddings(shows_titles, shows_descriptions):
    """
    Generates embeddings for a list of TV shows based on their descriptions.

    Parameters:
        shows_titles (list): A list of TV show titles.
        shows_descriptions (list): A list of descriptions corresponding to the TV show titles.

    Returns:
        dict: A dictionary of TV show embeddings in the format `{show name: vector}`.
    """
    client = connect_to_openai()
    dict_shows_vectors = {}
    for i in range(len(shows_titles)):
        response = client.embeddings.create(
            input=shows_descriptions[i],
            model="text-embedding-3-small"
        )
        dict_shows_vectors[shows_titles[i]] = response.data[0].embedding

    return dict_shows_vectors


