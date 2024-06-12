import os
import pickle

from pykeen.triples import TriplesFactory


def convert(data_to_convert, recreate=True):

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
