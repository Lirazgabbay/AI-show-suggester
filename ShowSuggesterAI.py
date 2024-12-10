# showSuggesterAI.py
import pandas as pd
from thefuzz import process

df = pd.read_csv('imdb_tvshows.csv')

def ask_from_user():
    # ask from user for N shows and return the list of shows after fixing typos and matching to csv file
    # while no -> ask again from user -> each step call match_showes() and confirm_matches()
    # call the match shows function and confirm matches function
    pass

def fix_and_match_shows(user_shows):
    # return a list of shows that match the user's shows based on csv file using fuzzy matching
    tv_shows = df['Title'].tolist()
    raw_show_list = user_shows.split(',')
    user_shows_list = [show.strip() for show in raw_show_list]

    matched_shows = []
    for show in user_shows_list:
        # Use fuzzy matching to find the closest show title
        match = process.extractOne(show, tv_shows)
        # process.extractOne returns a tuple with the matched show title and the similarity score
        if match and match[1] > 70:  # Only consider matches with a confidence score > 70
            matched_shows.append(match[0])

    return matched_shows

def confirm_matches(fixed_shows_names):
    # ask the user for confirmation after fixing shows names and return True if confirmed
    pass

def generate_embeddings(shows):
    # return embedding vector for all shows in csv file
    pass

def save_embedding_to_pickle(user_embedding):
    # save the  generate_embeddings to a pickle file such that we will call the embeddings API 
    # of open ai once on each of the shows’ descriptions
    pass

def pickle_hit_or_miss():
    # check if the pickle file exists and return True if it does
    pass

def load_embedding_from_pickle(shows_names_list):
    # load the embeddings from the pickle file
    pass

def generate_average_embedding(user_embedding):
    # return the average embedding of the user's shows
    # 1. pickle_hit_or_miss
    # 2. if hit - load_embedding_from_pickle, else- generate_embeddings and save_embedding_to_pickle
    # 3. generate_avarage_embedding
    pass

def distances_between_embeddings(average_user_embedding, all_embeddings):
    # return dictionary: showname -> distance
    # the distance is between average user's embedding to all tv shows embeddings
    # later : use usearch or annoy to find the closest shows to the user's embedding
    pass

def closest_shows(distance_dict):
    # receive dictionary: showname -> distance_to_avg_TV_shows_embedding 
    # return dict of the top 5 closest shows to the user's embedding sorting by distance (shortest distance first)
    # the return value will be names
    pass

def converte_to_percentages(closest_shows):
    # receive dictionary of the top 5 closest shows: showname -> distance_to_avg_TV_shows_embedding 
    # return a list of shows with their percentages similarity to the user's embedding
    pass

def genrate_new_recommendations(user_embedding, distances_between_average_all_embeddings):
    # return a list of shows with their cosine similarity to the user's embedding
    pass

def print_recommendations(recommendations, percentages):
    # print the recommendations
    pass

def main():
    fixed_user_input = ask_from_user()
    avg_embedding = generate_average_embedding(fixed_user_input)
    all_embeddings = load_embedding_from_pickle(df['Title'] ) #csv['Title'] 
    user_embedding = load_embedding_from_pickle(fixed_user_input)
    dict_distances = distances_between_embeddings(avg_embedding,all_embeddings)
    closest_shows = closest_shows(dict_distances)
    percentages = converte_to_percentages(closest_shows)
    recommendations = genrate_new_recommendations(user_embedding, dict_distances)
    print_recommendations(recommendations, percentages)
