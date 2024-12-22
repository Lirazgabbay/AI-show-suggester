import requests
from PIL import Image
from io import BytesIO
import time
import json
from AI_models import connect_to_LightX
from embeddings import connect_to_openai


def generate_new_tv_shows(user_shows_list, user_shows_descriptions_list, recomended_shows, recomended_shows_descriptions):
    """
    Generates and displays advertisements for two new TV shows based on user preferences 
    and recommended shows.
    """
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
    """Prints show's descriptions"""
    print(f"I have also created just for you two shows which I think you would love.\n"
        f"Show #1 is based on the fact that you loved the input shows that you gave me. "
        f"Its name is {show1name} and it is about {show1description}\n"
        f"Show #2 is based on the shows that I recommended for you. Its name is {show2name} "
        f"and it is about {show2description}\n"
        f"Here are also the 2 tv show ads. Hope you like them!")
    

def generate_tv_show_ad(new_show_name, new_show_description):
    """
    Generates an image advertisement for a TV show based on its name and description using LightX API.
    This function sends a request to the LightX API to create an image inspired by the given TV show 
    name and description. It handles the asynchronous nature of the API by polling the status of 
    the image generation.

    Args:
        new_show_name (str): The name of the new TV show.
        new_show_description (str): A brief description of the new TV show.

    Returns:
        str or None: The URL of the generated image if successful, or None if the image generation fails.
    """
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




