from pykeen.predict import predict_triples, predict_target


def predict_for_user(result, data, username):
    print('Start Predicting ...')

    pack = predict_triples(model=result.model, triples=data)
    df = pack.process(factory=result.training).df

    pred = predict_target(
        model=result.model,
        head=username,
        relation="like",
        triples_factory=result.training,
    )
    pred_filtered = pred.filter_triples(result.training)

    df = pred_filtered.df

    print(f'Here are some predictions for user {username}: ')
    print(df.nlargest(n=10, columns="score")['tail_label'].to_list())