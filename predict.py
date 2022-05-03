import numpy as np
import pandas as pd

from model import Model
from util import get_imdb_data

from config import (
    PREDICT_IMDB_CSV
)


def predict(take_samples=False):
    model = Model()
    data_df = get_imdb_data(load_data=True)
    predict_df = data_df.copy()
    predict_df['prediction'] = pd.NA

    if take_samples:
        # Sample five reviews
        predict_df = predict_df.sample(
                                        # n=5,
                                        random_state=1,
                                        # random_state=50,
                                        frac=0.0001
                                        )
        # .reset_index(drop=True)
    for idx, row in predict_df.iterrows():
        predict_df.at[idx, 'prediction'] = ' '.join(model.predict(row['masked']))
    print(predict_df)

    predict_df.to_csv(PREDICT_IMDB_CSV)


if __name__ == '__main__':
    predict()
