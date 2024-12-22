# showSuggesterAI.py
from data_handler import load_csv
from embeddings import pickle_hit_or_miss
from generative_show import generate_new_tv_shows
from recomendations import generate_average_embeddings, genrate_new_recommendations
from user_interaction import ask_from_user


def get_shows_descriptions(shows_list):
    """
    This function looks up each show title in the provided list within a CSV file (loaded via `load_csv`)
    and retrieves its description.

    Args:
        shows_list (list): A list of show titles (str) for which descriptions are to be retrieved.

    Returns:
        list: A list of descriptions (str) corresponding to the input show titles.
    """
    df = load_csv()
    shows_descriptions = []
    for show in shows_list:
        description = df[df['Title'] == show]['Description'].values[0]
        shows_descriptions.append(description)
    return shows_descriptions


def main():
    fixed_user_input = ask_from_user()
    dict_shows_vectors = pickle_hit_or_miss()
    avg_embedding = generate_average_embeddings(fixed_user_input, dict_shows_vectors)
    closest_shows_list = genrate_new_recommendations(fixed_user_input, avg_embedding, dict_shows_vectors)
    user_shows_descriptions_list = get_shows_descriptions(fixed_user_input)
    recomended_shows_descriptions = get_shows_descriptions(closest_shows_list)
    generate_new_tv_shows(fixed_user_input,user_shows_descriptions_list ,closest_shows_list,recomended_shows_descriptions)
    

if __name__ == "__main__":
    main()
   
