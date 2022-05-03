import numpy as np
import pandas as pd

from model import Model
from util import get_imdb_data


def predict():
    model = Model()
    data_df = get_imdb_data()
    data_df['prediction'] = pd.NA
    # Sample five reviews
    sample = data_df.sample(
                            # n=5,
                            random_state=50,
                            frac=0.0001
                            ).reset_index(drop=True)
    for idx, row in sample.iterrows():
        sample.at[idx, 'prediction'] = ' '.join(model.predict(row['masked']))
    print(sample)


if __name__ == '__main__':
    predict()