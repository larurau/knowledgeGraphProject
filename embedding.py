import inspect
import os
import pickle

import pykeen.models
from pykeen.evaluation import RankBasedEvaluator
from pykeen.pipeline import pipeline


def evaluate_model(training_result, model_name):

    print(f"Evaluating trained {model_name} model:")
    print(f"Hits@1: {training_result.metric_results.get_metric('hits@1')}")
    print(f"Hits@3: {training_result.metric_results.get_metric('hits@3')}")
    print(f"Hits@5: {training_result.metric_results.get_metric('hits@5')}")
    print(f"Hits@10: {training_result.metric_results.get_metric('hits@10')}")
    print(f"Mean Reciprocal Rank: {training_result.metric_results.get_metric('mean_reciprocal_rank')}")
    print(f"Mean Rank: {training_result.metric_results.get_metric('mean_rank')}\n")


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
    print('Evaluating different embedding models:\n\n')

    all_entities = dir(pykeen.models)
    model_classes = [entity for entity in all_entities if inspect.isclass(getattr(pykeen.models, entity))]
    print(model_classes)

    # models_1 = ['AutoSF', 'BoxE', 'ComplEx', 'ConvE', 'ConvKB', 'CooccurrenceFilteredModel', 'CrossE', 'DistMA', 'ERMLP', 'FixedModel', 'HolE', 'KG2E', 'MuRE', 'NTN', 'PairRE', 'ProjE', 'QuatE', 'RESCAL', 'RGCN', 'RotatE', 'SE', 'SimplE', 'TorusE', 'TransD', 'TransE', 'TransF', 'TransH', 'TransR', 'TuckER', 'UM']
    # models_2 = ['MuRE', 'QuatE', 'HolE', 'CrossE', 'RGCN', 'ERMLP']
    models_3 = ['MuRE', 'QuatE', 'HolE', 'CrossE', 'ERMLP']

    performance_results = []

    for model in models_3:
        print(f'Evaluating model: {model}')
        trained = train_model(triples_factory, model)
        evaluate_model(trained, model)
        performance_results.append((trained.metric_results.get_metric('mean_rank'), model))
        print('\n                        *******************                        \n')

    print('*******************************************************************')

    sorted_performance_results = sorted(performance_results, key=lambda x: x[0])

    with open('output\\log.txt', 'w') as file:
        for performance_metric, model_name in sorted_performance_results:
            print(f'{performance_metric} mean rank - {model_name}', file=file)
