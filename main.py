import os
import pickle

from pykeen.evaluation import RankBasedEvaluator
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

from dataPreparation import import_data


def convert_to_triples(data_to_convert, recreate=True):

    path = 'output/triplesFactory.pkl'
    if not recreate and os.path.exists(path):
        print("Loading TriplesFactory from file...\n")
        with open(path, 'rb') as file:
            return pickle.load(file)

    print('Start Converting Data ...')

    data_to_convert = data_to_convert.astype(str)

    triples_converted = data_to_convert[['Subject', 'Predicate', 'Object']].values

    triple_factory = TriplesFactory.from_labeled_triples(triples_converted)

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

    evaluator = RankBasedEvaluator(batch_size=16, automatic_memory_optimization=True)

    result = pipeline(
        model='TransE',
        loss="softplus",
        random_seed=42,
        training=training,
        testing=testing,
        validation=validation,
        model_kwargs=dict(
            embedding_dim=50,
            random_seed=42
        ),
        optimizer_kwargs=dict(
            lr=0.003
        ),
        training_kwargs=dict(
            num_epochs=50,
            use_tqdm_batch=False
        ),
        evaluator=evaluator
    )

    with open(path, 'wb') as file:
        pickle.dump(result, file)
    return result


if __name__ == '__main__':

    reimportData = True
    oneThousandthOfTriples = 10
    importedData = import_data(oneThousandthOfTriples, reimportData)

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
