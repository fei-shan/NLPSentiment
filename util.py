import string
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

from config import (
    LOCAL_DATA_DIR,
    IMDB_CSV,
)


def get_imdb_data():
    imdb_df = pd.read_csv(IMDB_CSV)
    imdb_df['tokenized'] = preprocess(imdb_df['review'])
    print(imdb_df)
    return imdb_df


def preprocess(series):
    remove_words = {'br', 'nt'}
    stop_words = set(stopwords.words('english'))
    stop_words.update(remove_words)

    punctuation_table = str.maketrans('', '', string.punctuation)
    tokenized = series.copy()
    max_len = 0
    for i, s in series.items():
        # convert to lower case
        s = s.lower()
        # nltk tokenize
        tokens = word_tokenize(s)
        # remove punctuation from each token
        tokens = [w.translate(punctuation_table) for w in tokens]
        # remove stopwords
        tokens = [w for w in tokens if w not in stop_words]
        # remove hanging 's' and 'a'
        tokens = [w for w in tokens if len(w) > 1]
        # remove tokens with numbers in them
        tokens = [w for w in tokens if w.isalpha()]
        # # append starting and ending, store as string
        # tokens = 'startseq ' + ' '.join(tokens) + ' endseq'
        # get max length
        if len(tokens) > max_len: max_len = len(tokens)
        # add to new pandas series
        tokenized[i] = tokens
    print(f'max length of tokens: {max_len}')
    return tokenized
    # return series.apply(word_tokenize)


if __name__ == '__main__':
    get_imdb_data()
