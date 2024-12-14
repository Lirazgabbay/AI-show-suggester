from ShowSuggesterAI import distances_between_embeddings,fix_and_match_shows,generate_average_embeddings, genrate_new_recommendations, load_embedding_from_pickle, closest_shows
import pytest
import pandas as pd
import numpy as np

# Sample data
df = pd.read_csv('imdb_tvshows.csv')

# Test 1: Check if the user enter "user_input" fix_and_match_shows return "expected_fixed_output"
def test_ask_from_user():
    user_shows = ["gem of throns", "lupan", "witcher"]
    expected_fixed_output = ["Game of Thrones", "Lupin", "The Witcher"]
    # Call the function and get the result
    result = fix_and_match_shows(user_shows)
    print(result)
    print(expected_fixed_output)
    assert result == expected_fixed_output

# Test 2: Check if the user enter unknown shows fix_and_match_shows return []
def test_match_shows_no_matches():
    user_shows = ["unknown show", "random series"]
    expected_matches = []  
    result = fix_and_match_shows(user_shows)
    assert result == expected_matches

# Test 3: Check if the user enter unknown show and a user input 
def test_match_shows_partially_matches():
    user_shows = ["unknown show", "lupan"]
    expected_matches = ["Lupin"]  
    result = fix_and_match_shows(user_shows)
    assert result == expected_matches

# Test 4: check for matching shows
def test_match_shows_perfect_matches():
    user_shows = ["Lupin"]
    expected_matches = ["Lupin"]  
    result = fix_and_match_shows(user_shows)
    assert result == expected_matches

# Test 5: check for no reapiting match shows
def test_match_shows_repeat_matches():
    user_shows = ["Luin", "Lupn"]
    expected_matches = ["Lupin"]  
    result = fix_and_match_shows(user_shows)
    assert result == expected_matches

# Test 6: run an example of distance_between_embeddings
def test_distance_between_embeddings():
    # Test case with valid embeddings
    average_user_embedding = np.array([0.5, 0.2, 0.8])
    all_embeddings = {
        "Game of Thrones": np.array([0.6, 0.1, 0.7]),
        "Lupin": np.array([0.1, 0.3, 0.9]),
        "The Witcher": np.array([0.5, 0.2, 0.8]),
    }
    # Expected distances calculated manually
    expected_distances = {
        "Game of Thrones": np.linalg.norm(average_user_embedding - all_embeddings["Game Of Thrones"]),
        "Lupin": np.linalg.norm(average_user_embedding - all_embeddings["Lupin"]),
        "The Witcher": np.linalg.norm(average_user_embedding - all_embeddings["The Witcher"]),
    }
    result = distances_between_embeddings(average_user_embedding, all_embeddings)
    assert result == pytest.approx(expected_distances)

# Test 7: run an example of distance_between_embeddings to check if the function handles empty embeddings
def test_distance_between_embeddings_empty_embeddings():
    average_user_embedding = np.array([0.5, 0.2, 0.8])
    all_embeddings = {} 
    expected_distances = {} 
    result = distances_between_embeddings(average_user_embedding, all_embeddings)
    assert result == expected_distances

# Test 8: run an example of distance_between_embeddings to check if the function handles empty average_user_embedding
def test_distance_between_embeddings_identical_embeddings():
    average_user_embedding = np.array([0.5, 0.2, 0.8])
    all_embeddings = {
        "Game of Thrones": np.array([0.5, 0.2, 0.8])
    }
    expected_distances = {
        "Game of Thrones": 0.0
    }
    result = distances_between_embeddings(average_user_embedding, all_embeddings)
    assert result == expected_distances

# Test 9: Validates the function's behavior with a large dataset, ensuring all distances are calculated and non-negative.
def test_distance_between_embeddings_large_input():
    average_user_embedding = np.random.rand(100)
    all_embeddings = {f"Show {i}": np.random.rand(100) for i in range(1000)}
    result = distances_between_embeddings(average_user_embedding, all_embeddings)
    assert len(result) == 1000
    assert all(distance >= 0 for distance in result.values())

# Test 10: if the user enter "Lupin" the reccomendtions won't include "Lupin"
def test_genrate_new_recommendations_multiple_options():
    fixed_user_input = ["Lupin", "The Witcher"]
    avg_embedding = generate_average_embeddings(fixed_user_input)
    all_embeddings = load_embedding_from_pickle(df['Title'] ) #csv['Title'] 
    user_embedding = load_embedding_from_pickle(fixed_user_input)
    dict_distances = distances_between_embeddings(avg_embedding,all_embeddings)
    closest_shows = closest_shows(dict_distances)
    recommendations = genrate_new_recommendations(user_embedding, dict_distances)
    assert ["Lupin", "The Witcher"] not in recommendations

# Test 11: if the user enter "Lupin" the reccomendtions won't include "Lupin"
def test_genrate_new_recommendations_one_option():
    fixed_user_input = ["Lupin"]
    avg_embedding = generate_average_embeddings(fixed_user_input)
    all_embeddings = load_embedding_from_pickle(df['Title'] ) #csv['Title'] 
    user_embedding = load_embedding_from_pickle(fixed_user_input)
    dict_distances = distances_between_embeddings(avg_embedding,all_embeddings)
    closest_shows = closest_shows(dict_distances)
    recommendations = genrate_new_recommendations(user_embedding, dict_distances)
    assert "Lupin" not in recommendations