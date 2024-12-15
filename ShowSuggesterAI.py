# showSuggesterAI.py
import numpy as np
import pandas as pd
from thefuzz import process
import os
import pickle
from openai import OpenAI
import requests
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import time
import openai
import json


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

    user_tv_shows = []
    while True:
        print(f"Enter a TV show (or press Enter to finish): ")
        show = input()
        if not show:  # Stop if Enter is pressed
            break
        user_tv_shows.append(show.strip())
        print(f"Added '{show.strip()}' to your list!")
    return user_tv_shows


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
    if not fixed_shows_names:
        print("No matches found.")
        return False
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
    dict_show_distance = distances_embeddings_avg(avg_embedding, dict_shows_vectors)
    closest_shows_dict = closest_shows(user_input, dict_show_distance)
    dict_show_percentages = converte_to_percentages(closest_shows_dict, dict_show_distance)
    print("Here are the tv shows that I think you would love:")
    for show in closest_shows_dict:
        print(f"{show} ({dict_show_percentages[show]}%)")
    return list(closest_shows_dict.keys())

def distances_embeddings_avg(avg_user_embedding, dict_shows_vectors):
    # return dictionary: showname -> distance from avg
    # later : use usearch or annoy to find the closest shows to the user's embedding
    dict_show_distance = {}
    for show, vector in dict_shows_vectors.items():
        avg_user_embedding = np.array(avg_user_embedding)
        vector = np.array(vector)
        distance = np.linalg.norm(avg_user_embedding - vector)
        dict_show_distance[show] = distance

    return dict_show_distance
    

def closest_shows(user_input, distance_dict, top_n=5):
    # receive dictionary: showname -> distance_to_avg_TV_shows_embedding 
    # return dict of the top n = 5 closest shows (different from user_input) to the user's embedding sorting by distance (shortest distance first)
    # the return value will be names
    filtered_dict = {}
    # filter out the shows that the user already input
    for show_name, distance_to_avg in distance_dict.items():
        if show_name not in user_input:
            filtered_dict[show_name] = distance_to_avg
    sorted_top_n = sorted(filtered_dict.items(), key=lambda item: item[1], reverse=False)[:top_n]
    # Convert Back to Dictionary
    dict_closestShow_distance = dict(sorted_top_n)
    return dict_closestShow_distance

def converte_to_percentages(closest_shows_dict, dict_show_distance ):
    # receive dictionary of the top 5 closest shows: showname -> distance_to_avg_TV_shows_embedding 
    # return a list of shows with their percentages similarity to the user's embedding
    max_distance = max(dict_show_distance.values()) #retrive the max distance
    min_distance = min(closest_shows_dict.values())


    if max_distance == min_distance:
        return {show_name: 100 for show_name in closest_shows_dict}
    else:
        dict_show_similarity = {}
        for show_name, distance_to_avg in closest_shows_dict.items():
            normalized_distance = (distance_to_avg - min_distance) / (max_distance - min_distance)
            similarity = (1 - normalized_distance) * 99  # Invert normalized distance
            dict_show_similarity[show_name] = int(similarity)
        return dict_show_similarity
    

def generate_new_tv_shows(user_shows_list, user_shows_descriptions_list, recomended_shows, recomended_shows_descriptions):
    # last step - generate 2 new shows ad according to user's input and program recommendations for him
    # prints the a message and display the images to the screen
    new_show_name1 , new_description_name1 = generate_showName_description(user_shows_list, user_shows_descriptions_list)
    new_show_name2 , new_description_name2 = generate_showName_description(recomended_shows, recomended_shows_descriptions)
    url_1 = generate_tv_show_ad(new_show_name1 , new_description_name1)
    url_2 = generate_tv_show_ad(new_show_name2 , new_description_name2)
    print_show_ad(new_show_name1, new_description_name1, new_show_name2, new_description_name2)

    if url_1:
        open_image_from_url(url_1)
    else:
        print("Failed to generate image for first show")
        
    if url_2:
        open_image_from_url(url_2)
    else:
        print("Failed to generate image for second show")


def generate_showName_description(shows, descriptions):
    """
    Connects to OpenAI to generate a new TV show name and description based on given shows.

    Args:
        shows (list of str): List of TV show names.
        descriptions (list of str): Corresponding descriptions of the TV shows.

    Returns:
        tuple: (new_show_name, new_show_description)
    """
    client = connect_to_openai()

    # Combine shows and descriptions into a single string
    tvShows_description = "\n".join(
        [f"TV Show Name: {show}, Description: {description}" for show, description in zip(shows, descriptions)]
    )

    # Define the messages for the chat completion
    messages = [
        {
            "role": "system", 
            "content": """You are an expert screenwriter who creates TV shows based on content the user likes. 
            Create a new show with a title and a complete two-sentence description.
            You must respond using this exact format, including the curly braces and quotes:
            {
                "title": "Your Show Title Here",
                "description": "Your complete two-sentence description here."
            }"""
        },
        {
            "role": "user", 
            "content": f"Based on these shows, suggest a new imaginary show:\n{tvShows_description}"
        }
    ]

    try:
        # Generate a response from the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,  # Add some creativity
            max_tokens=300    # Ensure we get a complete response
        )
        
        # Get the response content
        response_text = response.choices[0].message.content.strip()
        try:
            result = json.loads(response_text)

            title = result.get("title", "").strip()
            description = result.get("description", "").strip()

            if not title or not description:
                print("Error: Generated response is missing title or description")
                print(f"Received response: {response_text}")
                return None, None
            
            return title, description

        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            print(f"Received response: {response_text}")
            return None, None

    except Exception as e:
        print(f"An error occurred while generating the show name and description: {e}")
        return None, None


def open_image_from_url(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Check if the request was successful
        image = Image.open(BytesIO(response.content))
        image.show()
    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve image from {image_url}: {e}")
    except Exception as e:
        print(f"Error processing image from {image_url}: {e}")


def print_show_ad(show1name, show1description, show2name, show2description):
    # Print the show descriptions
    print(f"I have also created just for you two shows which I think you would love.\n"
        f"Show #1 is based on the fact that you loved the input shows that you gave me. "
        f"Its name is {show1name} and it is about {show1description}\n"
        f"Show #2 is based on the shows that I recommended for you. Its name is {show2name} "
        f"and it is about {show2description}\n"
        f"Here are also the 2 tv show ads. Hope you like them!")
    

def generate_tv_show_ad(new_show_name, new_show_description):
    url = 'https://api.lightxeditor.com/external/api/v1/text2image'
    api_key = connect_to_LightX()
    headers = {
        'Content-Type': 'application/json',
        'x-api-key': api_key
    }

    # Create the data payload with both the show name and description as the text prompt
    data = {
        "textPrompt": f"An image inspired by the TV show '{new_show_name}' and the Description: {new_show_description}"
    }

    # Send request to generate image
    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        # Retrieve the orderId for status checking
        order_id = response.json()['body']['orderId']
        # Now check the status of the image generation
        check_url = 'https://api.lightxeditor.com/external/api/v1/order-status'
        status_payload = {
            "orderId": order_id
        }

        retries = 0
        max_retries = 5
        status = "init"
        image_url = None

        # Keep checking the status until the image is ready or retries are exhausted
        while status != "active" and retries < max_retries:
            status_response = requests.post(check_url, headers=headers, json=status_payload)

            if status_response.status_code == 200:
                status_info = status_response.json()['body']
                status = status_info['status']
                if status == "active":
                    return status_info.get('output')  # This is the image URL
            else:
                print(f"Failed to check status. Status code: {status_response.status_code}")
                break

            # Wait for 3 seconds before checking again
            time.sleep(3)
            retries += 1

        if image_url:
            # You can use the image URL to download or display the image
            image_response = requests.get(image_url)
            if image_response.status_code == 200:
                img = Image.open(BytesIO(image_response.content))
                return img
            else:
                print("Image generation failed or was not completed in time.")
                return None
    else:
        return None



def connect_to_LightX():
    # Connect to the LightX and return the api-key
    # Get API key from environment variable
    load_dotenv()
    api_key = os.getenv('LightX_API_KEY')
    if not api_key:
        raise ValueError("LightX_API_KEY environment variable is not set. Please set it with your LightX key.")
    print(f"API Key loaded: {api_key[:5]}...") # Print first 5 chars to verify it's loaded
    return api_key


def get_shows_descriptions(shows_list):
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
    closest_shows = genrate_new_recommendations(fixed_user_input, avg_embedding, dict_shows_vectors)
    user_shows_descriptions_list = get_shows_descriptions(fixed_user_input)
    recomended_shows_descriptions = get_shows_descriptions(closest_shows)
    generate_new_tv_shows(fixed_user_input,user_shows_descriptions_list ,closest_shows,recomended_shows_descriptions)
    
if __name__ == "__main__":
    main()
   
