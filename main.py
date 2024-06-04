from pykeen.pipeline import pipeline

if __name__ == '__main__':

    print('PyCharm')

    pipeline_result = pipeline(
        dataset='Nations',
        model='TransE',
    )

    print(f'Result: {pipeline_result}')