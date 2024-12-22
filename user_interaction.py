from thefuzz import process
from data_handler import load_csv


def ask_from_user():
    """
    Interacts with the user to collect a list of N TV shows, correct typos, and validate matches with the csv file.
    
    Returns:
        list: A list of corrected and validated TV show names provided by the user.
    """ 
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
    """Collects a list of TV show names entered by the user."""
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
    """
    Matches user-provided TV show names to the closest matches in the csv dataset using fuzzy matching.

    Parameters:
        user_shows (list): A list of TV show names provided by the user.

    Returns:
        list: A list of matched TV show names from the dataset.
    """
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
    """
    Validates the user input for TV show names.

    Parameters:
        user_input (list): A list of TV show names entered by the user.

    Returns:
        bool: 
            - `True` if at least one valid TV show name is provided.
            - `False` if all entries are empty or contain only whitespace.
    """
    shows = [show.strip() for show in user_input]
    if all (show == "" for show in shows):
        print("You didn't enter any TV shows.")
        return False
    return True
    

def confirm_matches(fixed_shows_names):
    """
    Asks the user for confirmation after fixing TV show names.

    Parameters:
        fixed_shows_names (list): A list of corrected TV show names.

    Returns:
        bool: 
            - `True` if the user confirms the matches.
            - `False` if the user denies or no matches are found.
    """
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

