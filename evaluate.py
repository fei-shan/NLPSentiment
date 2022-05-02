from model import Model
from util import get_imdb_data


def predict():
    model = Model()
    data_df = get_imdb_data()
    # sample five reviews
    sample = data_df.sample(frac=0.0001).reset_index(drop=True)
    print(sample)
    for idx, row in sample.iterrows():
        model.predict(row['masked'])


if __name__ == '__main__':
    predict()