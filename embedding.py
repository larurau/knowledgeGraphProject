import os
import pickle

from pykeen.evaluation import RankBasedEvaluator
from pykeen.pipeline import pipeline


def evaluate_model(training_result, model_name):
    print(f"Evaluating trained {model_name} model:")
    print(f"Hits@1: {training_result.metric_results.get_metric('hits@1')}")
    print(f"Hits@3: {training_result.metric_results.get_metric('hits@3')}")
    print(f"Hits@5: {training_result.metric_results.get_metric('hits@5')}")
    print(f"Hits@10: {training_result.metric_results.get_metric('hits@10')}")
    print(f"Mean Reciprocal Rank: {training_result.metric_results.get_metric('mean_reciprocal_rank')}")


def train_model(triples_factory, model_name, recreate=True):
    path = 'output/trainedModel.pkl'
    if not recreate and os.path.exists(path):
        print("Loading Trained Model...\n")
        with open(path, 'rb') as file:
            return pickle.load(file)

    print('Start Training Model ...')

    training, validation, testing = triples_factory.split(ratios=(.6, .2, .2), random_state=42)

    evaluator = RankBasedEvaluator(batch_size=16, automatic_memory_optimization=True)

    result = pipeline(
        model=model_name,
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


def compare_models(triples_factory):
    print('*******************************************************************')
    print('Evaluating different embedding models:')

    model_names = ['TransE', 'ComplEx', 'ConvE', 'TorusE']

    for model in model_names:
        print(f'Evaluating model: {model_names}')
        trained = train_model(triples_factory, model)
        evaluate_model(trained, model_names)
        print('\n                        *******************                        \n')

    print('*******************************************************************')
