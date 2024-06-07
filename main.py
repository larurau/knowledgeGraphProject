import os
import pandas
import pickle

from pykeen.evaluation import RankBasedEvaluator
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

def import_data(thousandths, reimport=True, random_reduction=False):

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
    print(f'Size before reduction is: {games_df.shape[0]}')
    factor = thousandths / 1000
    if random_reduction:
        print("The reduction is achieved by random sampling")
        games_df = games_df.sample(frac=factor, random_state=42)
    else:
        print("The reduction is achieved by taking the first elements of the dataframe")
        x = int(len(games_df) * factor)
        games_df = games_df.head(x)
    print(f'Size after reduction is: {games_df.shape[0]}')

    # Merge the dataframes on BGGId and select the necessary columns
    merged_df = pandas.merge(user_ratings_df, games_df, on='BGGId')[['Username', 'Rating', 'Name']]
    merged_df = merged_df.sort_values(by='Name')

    # Calculate the average rating
    median_rating = user_ratings_df['Rating'].median()
    print(f'Median is {median_rating} which is used for turning ratings into binary value')

    # Replace 'Rating' with 'likes' if above median, 'does not like' if below or equal to the median
    merged_df['Rating'] = merged_df['Rating'].apply(lambda x: 'likes' if x >= median_rating else 'does not like')

    print(f'Size of merged is: {merged_df.shape[0]}')
    only_likes = merged_df[merged_df['Rating'] == 'likes']
    print(f'Size of merged after only considering likes is: {only_likes.shape[0]}')


    # Add mechanics
    print("Adding mechanics:")
    mechanics_csv_path = 'resources/mechanics.csv'
    mechanics_df = pandas.read_csv(mechanics_csv_path)
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

    #Combining all triples:
    print(f'Combining {only_likes.shape[0]}  like relations with {triples_df.shape[0]} mechanic relations')
    triples_df.columns = ['Subject', 'Predicate', 'Object']
    only_likes.columns = ['Subject', 'Predicate', 'Object']

    combined_df = pandas.concat([triples_df, only_likes], ignore_index=True)
    print(f'Combined dataframe has: {combined_df.shape[0]}')

    print('Finished Importing\n')

    with open(path, 'wb') as file:
        pickle.dump(combined_df, file)
    return combined_df


def convert_to_triples(data_to_convert, recreate=True):

    path = 'output/triplesFactory.pkl'
    if not recreate and os.path.exists(path):
        print("Loading TriplesFactory from file...\n")
        with open(path, 'rb') as file:
            return pickle.load(file)

    print('Start Converting Data ...')

    data_to_convert = data_to_convert.astype(str)

    triples = data_to_convert[['Subject', 'Predicate', 'Object']].values

    triple_factory = TriplesFactory.from_labeled_triples(triples)

    print(triple_factory)

    print('Finished Converting\n')

    with open(path, 'wb') as file:
        pickle.dump(triple_factory, file)
    return triple_factory


def train_model(triples_factory, recreate=True):

    path = 'output/trainedModel.pkl'
    if not recreate and os.path.exists(path):
        print("Loading Trained Model...\n")
        with open(path, 'rb') as file:
            return pickle.load(file)

    print('Start Training Model ...')

    training, validation, testing = triples_factory.split(ratios=(.6, .2, .2), random_state=42)
    evaluator = RankBasedEvaluator()

    result = pipeline(
        model='TransE',
        loss="softplus",
        random_seed=42,
        training=training,
        testing=testing,
        validation=validation,
        model_kwargs=dict(embedding_dim=3, random_seed=42),
        optimizer_kwargs=dict(lr=0.1),
        training_kwargs=dict(num_epochs=1000, use_tqdm_batch=False),
        evaluator=evaluator
    )

    with open(path, 'wb') as file:
        pickle.dump(result, file)
    return result


if __name__ == '__main__':

    reimportData = True
    oneThousandthOfTriples = 5
    randomReduction = False
    importedData = import_data(oneThousandthOfTriples, reimportData, randomReduction)

    recreateTriplesFactory = True
    triples = convert_to_triples(importedData, recreateTriplesFactory)

    recreateTraining = True
    training_result = train_model(triples, recreateTraining)

    # Print the metrics
    print("Evaluating trained model:")
    print(f"Hits@1: {training_result.metric_results.get_metric('hits@1')}")
    print(f"Hits@3: {training_result.metric_results.get_metric('hits@3')}")
    print(f"Hits@5: {training_result.metric_results.get_metric('hits@5')}")
    print(f"Hits@10: {training_result.metric_results.get_metric('hits@10')}")
    print(f"Mean Reciprocal Rank: {training_result.metric_results.get_metric('mean_reciprocal_rank')}")

    print("Finished training\n")