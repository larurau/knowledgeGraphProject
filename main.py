from pykeen.datasets import Nations
import pandas

def show_data_structure():
    dataset = Nations()
    print(f'Nations dataset: {dataset.training.mapped_triples[:5]}')

    entity_to_id = dataset.training.entity_to_id
    id_to_entity = {v: k for k, v in entity_to_id.items()}
    relation_to_id = dataset.training.relation_to_id
    id_to_relation = {v: k for k, v in relation_to_id.items()}

    # Print some of the entity and relation mappings
    print("\nSample entity mappings:")
    for entity, id_ in list(entity_to_id.items())[:5]:
        print(f"{id_}: {entity}")

    print("\nSample relation mappings:")
    for relation, id_ in list(relation_to_id.items())[:5]:
        print(f"{id_}: {relation}")

def import_data():

    print('Start Importing')

    games_csv_path = 'resources/games.csv'
    user_ratings_csv_path = 'resources/user_ratings.csv'

    # Read and select necessary columns from the CSV files
    games_df = pandas.read_csv(games_csv_path)[['BGGId', 'Name']]
    user_ratings_df = pandas.read_csv(user_ratings_csv_path)[['BGGId', 'Rating', 'Username']]

    print('Restructuring dataset:')

    # Merge the dataframes on BGGId and select the necessary columns
    merged_df = pandas.merge(user_ratings_df, games_df, on='BGGId')[['Username', 'Rating', 'Name']]
    print(f'Merged data has columns: {merged_df.columns}: ')

    # Calculate the average rating
    median_rating = user_ratings_df['Rating'].median()
    print(f'Median is {median_rating} which is used for turning ratings into binary value')

    # Replace 'Rating' with 'likes' if above median, 'does not like' if below or equal to the median
    merged_df['Rating'] = merged_df['Rating'].apply(lambda x: 'likes' if x >= median_rating else 'does not like')
    print('Example row:')
    print(merged_df.head(1))

    # Calculate the frequency of each rating after filtering
    rating_counts = merged_df['Rating'].value_counts()
    print("Frequency of each rating, as a lot of values fall directly on the median these values are not equal:")
    print(rating_counts)

    print('Finished Importing\n')


if __name__ == '__main__':

    import_data()
    show_data_structure()