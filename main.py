from embedding import train_model, evaluate_model, compare_models
from triplesConversion import convert
from dataPreparation import import_data


if __name__ == '__main__':

    reimportData = True
    oneThousandthOfTriples = 10
    importedData = import_data(oneThousandthOfTriples, reimportData)

    recreateTriplesFactory = True
    triples = convert(importedData, recreateTriplesFactory)

    recreateTraining = True

    training_result = train_model(triples, 'TransE', recreateTraining)
    evaluate_model(training_result, 'TransE')

    print("Finished training\n")
