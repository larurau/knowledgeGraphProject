import inspect
import os
import pickle

import pykeen.models
from pykeen.evaluation import RankBasedEvaluator
from pykeen.pipeline import pipeline


def evaluate_model(training_result, model_name, file):

    print(f"Evaluating trained {model_name} model:")
    print(f"Hits@1: {training_result.metric_results.get_metric('hits@1')}")
    print(f"Hits@3: {training_result.metric_results.get_metric('hits@3')}")
    print(f"Hits@5: {training_result.metric_results.get_metric('hits@5')}")
    print(f"Hits@10: {training_result.metric_results.get_metric('hits@10')}")
    print(f"Mean Reciprocal Rank: {training_result.metric_results.get_metric('mean_reciprocal_rank')}")
    print(f"Mean Rank: {training_result.metric_results.get_metric('mean_rank')}\n")

    if file is not None:
        print(f"Evaluating trained {model_name} model:", file=file)
        print(f"Mean Rank: {training_result.metric_results.get_metric('mean_rank')}\n", file=file)


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

    with open('output\\log.txt', 'w') as file:
        # model_names = [pykeen.models.TransE,pykeen.models.AutoSF,pykeen.models.PairRE,pykeen.models.TransF]

        all_entities = dir(pykeen.models)
        model_classes = [entity for entity in all_entities if inspect.isclass(getattr(pykeen.models, entity))]
        print(model_classes)

        all_models = ['AutoSF', 'BoxE', 'ComplEx', 'ConvE', 'ConvKB', 'CooccurrenceFilteredModel', 'CrossE', 'DistMA', 'DistMult', 'DistMultLiteral', 'DistMultLiteralGated', 'ERMLP', 'ERMLPE', 'ERModel', 'EvaluationOnlyModel', 'FixedModel', 'HolE', 'InductiveERModel', 'InductiveNodePiece', 'InductiveNodePieceGNN', 'KG2E', 'LiteralModel', 'MarginalDistributionBaseline', 'Model', 'MuRE', 'NTN', 'NodePiece', 'PairRE', 'ProjE', 'QuatE', 'RESCAL', 'RGCN', 'RotatE', 'SE', 'SimplE', 'SoftInverseTripleBaseline', 'TorusE', 'TransD', 'TransE', 'TransF', 'TransH', 'TransR', 'TuckER', 'UM']

        for model in all_models:
            print(f'Evaluating model: {model}')
            trained = train_model(triples_factory, model)
            evaluate_model(trained, model, file)
            print('\n                        *******************                        \n')

        print('*******************************************************************')
