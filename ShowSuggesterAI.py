# showSuggesterAI.py
import numpy as np
import pandas as pd
from thefuzz import process
import os
import pickle
from openai import OpenAI
from dotenv import load_dotenv


def load_csv():
    # load the csv file
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

def ask_from_user():
    # ask from user for N shows and return the list of shows after fixing typos and matching to csv file
    # return the correct list of shows requested by the user
    print("Welcome to the TV Show Suggester!")
    input_shows = collect_tv_shows()
    while(not valid_input(input_shows)):
        input_shows = collect_tv_shows() # ask for input again

    fixed_shows = fix_and_match_shows(input_shows)
    while not confirm_matches(fixed_shows):
        print("Sorry about that. Lets try again, please make sure to write the names of the tv shows correctly")
        input_shows = collect_tv_shows()
        fixed_shows = fix_and_match_shows(input_shows)
    print("Great! Generating recommendations now…")
    return fixed_shows

def collect_tv_shows():
    print("Enter the name of your favorite TV shows, one at a time.")
    print("When you're done, just press Enter without typing anything.")

    tv_shows = []
    while True:
        print(f"Enter a TV show (or press Enter to finish): ")
        show = input()
        if not show:  # Stop if Enter is pressed
            break
        tv_shows.append(show.strip())
        print(f"Added '{show.strip()}' to your list!")
    return tv_shows


def fix_and_match_shows(user_shows):
    # return a list of shows that match the user's shows based on csv file using fuzzy matching
    df = load_csv()
    tv_shows = df['Title'].tolist()
    user_shows_list = [show.strip() for show in user_shows] # remove leading and trailing whitespaces

    matched_shows = []
    seen_shows = set()

    for show in user_shows_list:
        if show is None or show == "":  # Skip empty strings
            continue
        # Use fuzzy matching to find the closest show title - process.extractOne returns a tuple with the matched show title and the similarity score
        match = process.extractOne(show, tv_shows)
        if match and match[1] > 70:  # Only consider matches with a confidence score > 70
            if match[0] not in seen_shows:
                matched_shows.append(match[0])
                seen_shows.add(match[0])
    return matched_shows


def valid_input(user_input):
    # check if the user input is valid    
    shows = [show.strip() for show in user_input]
    if all (show == "" for show in shows):
        print("You didn't enter any TV shows.")
        return False
    return True
    

def confirm_matches(fixed_shows_names):
    # ask the user for confirmation after fixing shows names and return True if confirmed
    str_fixed_shows_names = ', '.join(fixed_shows_names)
    print(f"Making sure, do you mean {str_fixed_shows_names}? (y/n)")
    user_input = input()
    while user_input.lower() not in ['y', 'n']: 
        print("invalid input, please enter 'y' for yes or 'n' for no.")
        user_input = input()
    if user_input.lower() == 'y':
        return True
    else:
        return False


def pickle_hit_or_miss():
    # check if the pickle file exists, otherwise create it, and return the dict_shows_vectors (show name: vector)
    if os.path.exists("tv_shows_embeddings.pkl"):
        return load_embedding_from_pickle()
    else:
        return save_new_embedding_to_pickle()
    

def load_embedding_from_pickle():
    # load the embeddings from the pickle file
    # deserialize the dictionary (show name: vector) and return it
    try:
        with open("tv_shows_embeddings.pkl", "rb") as pickle_file:
            deserialized_dict = pickle.load(pickle_file)
            return deserialized_dict
    except Exception as e:
        print(f"An error occurred while loading the embeddings from the pickle file: {e}")


def save_new_embedding_to_pickle():
    # save the generate_embeddings to a pickle file such that we will call the embeddings API 
    # of open ai once on each of the shows’ descriptions
    df = load_csv()
    dict_shows_vectors = generate_embeddings(df['Title'].tolist(), df['Description'].tolist())
    # serialize the dictionary to a pickle file
    try:
        with open("tv_shows_embeddings.pkl", "wb") as pickle_file:
            pickle.dump(dict_shows_vectors, pickle_file)
    except Exception as e:
        print(f"An error occurred while saving the embeddings to a pickle file: {e}")


def generate_embeddings(shows_titles, shows_descriptions):
    # return dict_shows_vectors: (show name: vector) for all shows in csv file if doesn't exist in pickle file
    client = connect_to_openai()
    dict_shows_vectors = {}
    for i in range(len(shows_titles)):
        response = client.embeddings.create(
            input=shows_descriptions[i],
            model="text-embedding-3-small"
        )
        dict_shows_vectors[shows_titles[i]] = response.data[0].embedding

    return dict_shows_vectors


def connect_to_openai():
    # connect to the openai API and return the client
    # Get API key from environment variable
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY_1')
    if not api_key:
        raise ValueError("OPENAI_API_KEY_1 environment variable is not set. Please set it with your OpenAI API key.")
    client = OpenAI(api_key=api_key)
    return client


def generate_average_embeddings(user_shows, dict_shows_vectors):
    # return the average embedding of the user's shows
    length = len(user_shows)
    user_embeddings = load_user_embedding(user_shows, dict_shows_vectors)

    # sum the vectors of all user's shows by cordination, and divide by the total number of user's shows
    average_vector = [sum(x)/length for x in zip(*user_embeddings)]

    return average_vector


def load_user_embedding(user_shows, dict_shows_vectors):
    user_embeddings = []
    for show in user_shows:
        current_vector = dict_shows_vectors[show]
        user_embeddings.append(current_vector)       

    return user_embeddings


def genrate_new_recommendations(user_input, avg_embedding, dict_shows_vectors):
    # return a list of shows with their closest similarity to the user's embedding
    pass

def distances_between_embeddings(avg_user_embedding, dict_shows_vectors):
    # return dictionary: showname -> distance from avg
    # later : use usearch or annoy to find the closest shows to the user's embedding
    dict_show_distance = {}
    for show, vector in dict_shows_vectors.items():
        distance = np.linalg.norm(avg_user_embedding - vector)
        dict_show_distance[show] = distance

    return dict_show_distance
    

def closest_shows(user_input, distance_dict):
    # receive dictionary: showname -> distance_to_avg_TV_shows_embedding 
    # return dict of the top 5 closest shows (different from user_input) to the user's embedding sorting by distance (shortest distance first)
    # the return value will be names
    pass

def converte_to_percentages(closest_shows):
    # receive dictionary of the top 5 closest shows: showname -> distance_to_avg_TV_shows_embedding 
    # return a list of shows with their percentages similarity to the user's embedding
    pass

def print_recommendations(recommendations, percentages):
    # print the recommendations
    pass


print(load_embedding_from_pickle()['Suits'])

def main():
    connect_to_openai()
    fixed_user_input = ask_from_user()
    dict_shows_vectors = pickle_hit_or_miss()
    avg_embedding = generate_average_embeddings(fixed_user_input, dict_shows_vectors)

    
    all_embeddings = load_embedding_from_pickle(df['Title'] ) #csv['Title'] 
    user_embedding = load_embedding_from_pickle(fixed_user_input)
    dict_distances = distances_between_embeddings(avg_embedding,all_embeddings)
    closest_shows = closest_shows(dict_distances)
    percentages = converte_to_percentages(closest_shows)
    recommendations = genrate_new_recommendations(user_embedding, dict_distances)
    print_recommendations(recommendations, percentages)
