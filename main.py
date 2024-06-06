import os

import numpy as np
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import pandas
import pickle

def import_data(percentage, reimport=True):

    path = 'output/importedData.pkl'
    if not reimport and os.path.exists(path):
        print("Loading imported data from file...\n")
        with open(path, 'rb') as file:
            return pickle.load(file)

    print('Start Importing')

    games_csv_path = 'resources/games.csv'
    user_ratings_csv_path = 'resources/user_ratings.csv'

    # Read and select necessary columns from the CSV files
    games_df = pandas.read_csv(games_csv_path)[['BGGId', 'Name']]
    user_ratings_df = pandas.read_csv(user_ratings_csv_path)[['BGGId', 'Rating', 'Username']]

    # Merge the dataframes on BGGId and select the necessary columns
    merged_df = pandas.merge(user_ratings_df, games_df, on='BGGId')[['Username', 'Rating', 'Name']]

    # Reduce the size of the dataset by 99%
    print(f'Size before reduction is: {merged_df.shape[0]}')
    merged_df = merged_df.sample(frac=percentage/100, random_state=42)
    print(f'Size after reduction is: {merged_df.shape[0]}')

    # Calculate the average rating
    median_rating = user_ratings_df['Rating'].median()
    print(f'Median is {median_rating} which is used for turning ratings into binary value')

    # Replace 'Rating' with 'likes' if above median, 'does not like' if below or equal to the median
    merged_df['Rating'] = merged_df['Rating'].apply(lambda x: 'likes' if x >= median_rating else 'does not like')

    only_likes = merged_df[merged_df['Rating'] == 'likes']
    print(f'Size after only considering likes is: {only_likes.shape[0]}')

    print('Finished Importing\n')

    with open(path, 'wb') as file:
        pickle.dump(only_likes, file)
    return only_likes


def convert_to_triples(data_to_convert, recreate=True):

    path = 'output/triplesFactory.pkl'
    if not recreate and os.path.exists(path):
        print("Loading TriplesFactory from file...\n")
        with open(path, 'rb') as file:
            return pickle.load(file)

    print('Start Converting Data')

    data_to_convert = data_to_convert.astype(str)

    triples = data_to_convert[['Username', 'Rating', 'Name']].values

    triple_factory = TriplesFactory.from_labeled_triples(triples)

    print('Finished Converting\n')

    with open(path, 'wb') as file:
        pickle.dump(triple_factory, file)
    return triple_factory

if __name__ == '__main__':

    reimportData = False
    recreateTriplesFactory = True
    percentageOfTriples = 1

    importedData = import_data(percentageOfTriples, reimportData)

    triples = convert_to_triples(importedData, recreateTriplesFactory)

    #training_result = train_model(tf, False)

    #print(training_result)

def train_model(triple_factory, recreate=True):

    path = 'output/trainedModel'

    if not recreate and os.path.exists(path):
        print("Loading Trained Model...")
        from pykeen.pipeline import PipelineResult
        return PipelineResult.load_from_directory(path)

    training, testing = triple_factory.split()

    result = pipeline(
        training=training,
        testing=testing,
        model='TransE',
        epochs=5
    )

    result.save_to_directory('output/test_pre_stratified_transe')
    return result