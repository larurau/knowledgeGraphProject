from embedding import train_model, evaluate_model, compare_models
from prediction import predict_for_user
from triplesConversion import convert
from dataPreparation import import_data


if __name__ == '__main__':

    reimportData = False
    oneTenThousandthOfTriples = 300
    importedData = import_data(oneTenThousandthOfTriples, reimportData)

    recreateTriplesFactory = False
    triples = convert(importedData, recreateTriplesFactory)

    # compare_models(triples)

    recreateTraining = False
    training_result = train_model(triples, 'QuatE', recreateTraining)
    evaluate_model(training_result, 'QuatE')

    predict_for_user(training_result, triples, "lionandlamb")

    print("Finished training\n")
