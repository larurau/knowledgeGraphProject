import math
import os
import pickle

import pandas


def reduce_like_triples_by_users(like_triples_df):
    user_counts = like_triples_df['Username'].value_counts()
    threshold = user_counts.quantile(0.20)
    frequent_users = user_counts[user_counts > threshold].index
    filtered_df = like_triples_df[like_triples_df['Username'].isin(frequent_users)]
    return filtered_df


def reduce_like_triples_by_games(like_triples_df):
    game_counts = like_triples_df['BGGId'].value_counts()
    threshold = game_counts.quantile(0.20)
    frequent_games = game_counts[game_counts > threshold].index
    filtered_df = like_triples_df[like_triples_df['BGGId'].isin(frequent_games)]
    return filtered_df


def reduce_like_triples(like_triples_df, target_factor):
    target_size = math.trunc(like_triples_df.shape[0] * target_factor)

    print(f'Trying to reduce the number of triples from {like_triples_df.shape[0]} to below {target_size}')
    print(f'Started with {like_triples_df["Username"].nunique()} users')
    print(f'Started with {like_triples_df["BGGId"].nunique()} games')

    i = 1
    while like_triples_df.shape[0] > target_size:
        print(f'Round {i}:')
        i = i + 1
        like_triples_df = reduce_like_triples_by_users(like_triples_df)
        print(f'   Reduced size to {like_triples_df.shape[0]} by excluding users')
        like_triples_df = reduce_like_triples_by_games(like_triples_df)
        print(f'   Reduced size to {like_triples_df.shape[0]} by excluding games')

    print(f'Ended with {like_triples_df["Username"].nunique()} users')
    print(f'Ended with {like_triples_df["BGGId"].nunique()} games')
    print('Sample of the resulting dataframes: ')
    user_counts = like_triples_df['Username'].value_counts()
    game_counts = like_triples_df['BGGId'].value_counts()
    print('The users with the most connections were:')
    print(user_counts.head(3))
    print('The users with the least connections were:')
    print(user_counts.tail(3))
    print('The games with the most connections were:')
    print(game_counts.head(3))
    print('The games with the least connections were:')
    print(game_counts.tail(3))
    print("-----------------------------")
    return like_triples_df


def import_data(part, reimport=True):
    path = 'output/importedData.pkl'
    if not reimport and os.path.exists(path):
        print("Loading imported data from file...\n")
        with open(path, 'rb') as file:
            return pickle.load(file)

    print('Start Importing ...')

    games_csv_path = 'resources/games.csv'
    user_ratings_csv_path = 'resources/user_ratings.csv'

    # Read and select necessary columns from the CSV files
    games_df = pandas.read_csv(games_csv_path)[['BGGId', 'Name']]
    user_ratings_df = pandas.read_csv(user_ratings_csv_path)[['BGGId', 'Rating', 'Username']]

    # Before merging reduce the size of the dataset
    factor = part / 10000
    print(f'Number of triples before reduction is {user_ratings_df.shape[0]}')
    user_ratings_df = reduce_like_triples(user_ratings_df, factor)
    print(f'Number of triples after reduction is {user_ratings_df.shape[0]}')

    # Merge the dataframes on BGGId and select the necessary columns
    merged_df = pandas.merge(user_ratings_df, games_df, on='BGGId')[['Username', 'Rating', 'Name']]
    merged_df = merged_df.sort_values(by='Name')

    # Calculate the average rating
    median_rating = user_ratings_df['Rating'].median()
    print(f'Median is {median_rating} which is used for turning ratings into binary value')

    # Replace 'Rating' with 'like' if above median, 'dislike' if below or equal to the median
    merged_df['Rating'] = merged_df['Rating'].apply(lambda x: 'like' if x >= median_rating else 'dislike')

    #print(f'Size of merged is: {merged_df.shape[0]}')
    #only_like = merged_df[merged_df['Rating'] == 'like']
    #print(f'Size of merged after only considering like is: {only_like.shape[0]}')

    # Add mechanics

    print("Adding mechanics:")
    mechanics_csv_path = 'resources/mechanics.csv'
    mechanics_df = pandas.read_csv(mechanics_csv_path)
    valid_bgg_ids = user_ratings_df['BGGId'].unique()
    mechanics_df = mechanics_df[mechanics_df['BGGId'].isin(valid_bgg_ids)]
    print(f'Size of mechanics is: {mechanics_df.shape[0]}')
    merged_mechanics_df = pandas.merge(games_df, mechanics_df, on='BGGId')
    print(f'Size of merged mechanics is: {merged_mechanics_df.shape[0]}')

    value_vars = [col for col in merged_mechanics_df.columns if col not in ['BGGId', 'Name']]
    melted_df = pandas.melt(merged_mechanics_df, id_vars=['Name'], value_vars=value_vars, var_name='Mechanic',
                            value_name='Indicator')

    # Filter for rows where Indicator is 1 (where the game has the mechanic)
    filtered_df = melted_df[melted_df['Indicator'] == 1].copy()
    # Add a fixed relationship column
    filtered_df['Relationship'] = 'hasMechanic'
    # Create the triples DataFrame
    triples_df = filtered_df[['Name', 'Relationship', 'Mechanic']]

    # Display the resulting DataFrame
    print(f'Split up the number of mechanic relations is: {triples_df.shape[0]}')

    # Combining all triples:
    print(f'Combining {merged_df.shape[0]}  like relations with {triples_df.shape[0]} mechanic relations')
    triples_df.columns = ['Subject', 'Predicate', 'Object']
    merged_df.columns = ['Subject', 'Predicate', 'Object']

    combined_df = pandas.concat([triples_df, merged_df], ignore_index=True)
    print(f'Combined dataframe has: {combined_df.shape[0]}')

    print('Finished Importing\n')

    with open(path, 'wb') as file:
        pickle.dump(combined_df, file)
    return combined_df
