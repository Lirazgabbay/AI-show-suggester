import numpy as np
from annoy import AnnoyIndex
from data_handler import load_csv

def generate_average_embeddings(user_shows, dict_shows_vectors):
    """
    Calculates the average embedding vector for a user's selected TV shows.

    Parameters:
        user_shows (list): A list of TV show names selected by the user.
        dict_shows_vectors (dict): A dictionary containing embeddings for TV shows in the format `{show name: vector}`.

    Returns:
        list: A list representing the average embedding vector for the user's selected shows.
    """
    length = len(user_shows)
    user_embeddings = load_user_embedding(user_shows, dict_shows_vectors)

    # sum the vectors of all user's shows by cordination, and divide by the total number of user's shows
    average_vector = [sum(x)/length for x in zip(*user_embeddings)]

    return average_vector


def load_user_embedding(user_shows, dict_shows_vectors):
    """Returns a list of embedding vectors corresponding to the user's selected TV shows."""
    user_embeddings = []
    for show in user_shows:
        current_vector = dict_shows_vectors[show]
        user_embeddings.append(current_vector)       

    return user_embeddings


def genrate_new_recommendations(user_input, avg_embedding, dict_shows_vectors):
    """
    Generates a list of TV show recommendations based on the user's input and average embedding.
    
    This function identifies the TV shows most similar to the user's embedding using an Annoy index.
    It returns the list of recommended show names and prints them with their similarity percentages.
    
    Args:
        user_input (str): The user's input, which could be used to refine recommendations.
        avg_embedding (list or np.ndarray): The user's average embedding vector.
        dict_shows_vectors (dict): A dictionary where keys are show names (str) and values
                                    are their corresponding embedding vectors (list or np.ndarray).
    
    Returns:
        list: A list of show names (str) that are closest to the user's embedding.
    """
    annoy_index, id_to_show = build_annoy_index(dict_shows_vectors)
    closest_shows_dict = closest_shows(user_input, avg_embedding, annoy_index, id_to_show, dict_shows_vectors)
    print("Here are the tv shows that I think you would love:")
    for show, percentage in closest_shows_dict.items():
        print(f"{show} ({percentage}%)")
    return list(closest_shows_dict.keys())


def build_annoy_index(dict_shows_vectors, metric='angular', n_trees=10):
    """
    This function builds an approximate nearest neighbor index using the Annoy library for a set of show embeddings. 
    It allows for fast similarity searches between shows based on their embeddings.

    args:
    dict_shows_vectors (dict): A dictionary mapping show names to their embeddings.
    metric (str): The distance metric to use. Default is 'angular'.
    n_trees (int): The number of trees to build in the index. Default is 10 (to balance between speed and accuracy).

    returns:
    AnnoyIndex: The built Annoy index.
    dict - id_to_show: A dictionary mapping index IDs to their corresponding show names.
    """
    # Determine embedding dimension
    if not dict_shows_vectors:
        raise ValueError("No embeddings found to build an index.")
    sample_vector = next(iter(dict_shows_vectors.values()))
    dimension = len(sample_vector)

    index = AnnoyIndex(dimension, metric=metric)
    id_to_show = {}
    
    i = 0
    for show_name, embedding in dict_shows_vectors.items():
        id_to_show[i] = show_name
        index.add_item(i, embedding)
        i += 1
    
    # Build the index
    index.build(n_trees)
    return index, id_to_show
    
    
def closest_shows(user_input, avg_embedding, annoy_index, id_to_show, dict_shows_vectors, top_n=5):
    """
    Finds the closest shows to a user's average embedding using Annoy index and calculates similarity scores.
    """
    candidate_count = top_n + len(user_input)
    neighbor_ids = annoy_index.get_nns_by_vector(avg_embedding, candidate_count)

    # Filter out shows already in user_input
    candidate_shows = [id_to_show[n_id] for n_id in neighbor_ids]
    filtered_candidates = [show for show in candidate_shows if show not in user_input]
    filtered_candidates = filtered_candidates[:top_n]
    
    filtered_vectors = {show: dict_shows_vectors[show] for show in filtered_candidates}
    top_n_distances_dict = distances_embeddings_avg(avg_embedding, filtered_vectors)
    similarity_scores = convert_to_percentages(top_n_distances_dict)
    
    return similarity_scores


def distances_embeddings_avg(avg_user_embedding, dict_shows_vectors):
    """
    Calculates the Euclidean distance between the user's average embedding and the embeddings of various shows.

    Args:
        avg_user_embedding (list or np.ndarray): The user's average embedding vector.
        dict_shows_vectors (dict): A dictionary where keys are show names (str) and values
                                    are their corresponding embedding vectors (list or np.ndarray).
    
    Returns:
        dict: A dictionary mapping each show name (str) to its distance (float) 
              from the user's average embedding.
    """
    dict_show_distance = {}
    for show, vector in dict_shows_vectors.items():
        avg_user_embedding = np.array(avg_user_embedding)
        vector = np.array(vector)
        distance = np.linalg.norm(avg_user_embedding - vector)
        dict_show_distance[show] = distance

    return dict_show_distance


def convert_to_percentages(top_n_distances_dict, n=5):
    """
    Converts distances to percentage similarity scores, only for the top N closest shows.

    Args:
        top_n_distances_dict (dict): A dictionary where keys are show names (str) and values are 
                                     their distances (float) to the user's average embedding.
        n (int): The number of top closest shows to consider for scaling the similarity scores. 
                 Default is 5.

    Returns:
        dict: A dictionary where keys are show names (str) and values are their similarity scores 
              (int, percentage) relative to the user's embedding.
    """

    min_dist = min(top_n_distances_dict.values())
    max_dist = max(top_n_distances_dict.values()) 
    
    df = load_csv()
    csv_length = len(df['Title'].tolist())
    print(csv_length)
    # Handle the case where all distances are the same
    if max_dist == min_dist:
        return {show: 100 for show in top_n_distances_dict.keys()}
    # Calculate similarity scores using a linear transformation
    similarity_scores = {}
    for show, dist in top_n_distances_dict.items():
        normalized_distance = (dist - min_dist) / (max_dist - min_dist)
        similarity = (1 - normalized_distance) 
        # scale the similarity base on lentdht of csv and n:
        min_scaling_factor = ((csv_length - n) / csv_length) * 100
        similarity = min_scaling_factor + similarity * (100 - min_scaling_factor)
        similarity_scores[show] = int(similarity)
    
    return similarity_scores