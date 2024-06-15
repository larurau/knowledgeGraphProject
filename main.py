from embedding import train_model, evaluate_model, compare_models
from prediction import predict_for_user
from triplesConversion import convert
from dataPreparation import import_data


if __name__ == '__main__':

    reimportData = False
    oneThousandthOfTriples = 10
    importedData = import_data(oneThousandthOfTriples, reimportData)

    recreateTriplesFactory = False
    triples = convert(importedData, recreateTriplesFactory)

    recreateTraining = False

    #compare_models(triples)
    training_result = train_model(triples, 'BoxE', recreateTraining)
    #evaluate_model(training_result, 'BoxE', None)

    predict_for_user(training_result, triples, "lionandlamb")

    print("Finished training\n")
