import os
import glob
import string
import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, MWETokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from datasets import load_dataset

from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer

# nltk.download('punkt')
# nltk.download('stopwords')

from config import (
    LOCAL_DATA_DIR,
    IMDB_50K_CSV,
    # IMDB_DATA_DIR,
    MASK,
    STEREOSET_TERMS,
    MASKED_IMDB_CSV
)


def get_imdb_data(model_checkpoint='distilbert-base-uncased', apply_mask=True, load_data=False):
    # imdb_df = pd.read_csv(IMDB_50K_CSV)
    # imdb_df['tokenized'] = preprocess(imdb_df['review'])

    if load_data:
        return pd.read_csv(MASKED_IMDB_CSV)

    imdb_dataset = load_dataset("imdb")
    train_df = imdb_dataset['train'].to_pandas()
    test_df = imdb_dataset['test'].to_pandas()
    imdb_df = train_df.append(test_df, ignore_index=True)
    # imdb_df['tokenized'] = preprocess(imdb_df['text'])
    print(imdb_df)

    bias_terms = set()
    multi_word_bias_terms = set()
    if apply_mask:
        for key, categories in STEREOSET_TERMS.items():
            for c in categories:
                for s in c:
                    split = s.split('_')
                    if len(split) > 1:
                        multi_word_bias_terms.add(s)
                    else:
                        bias_terms.add(s)
        # combine two sets
        bias_terms.update(multi_word_bias_terms)
        imdb_df['masked'], imdb_df['ground_truth'] = preprocess_mask(series=imdb_df['text'], mask_terms=bias_terms)
    print(imdb_df)

    # tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    masked_imdb_df = imdb_df[imdb_df['masked'].notnull()] if apply_mask else imdb_df
    masked_imdb_df.to_csv(MASKED_IMDB_CSV)
    return masked_imdb_df


def preprocess_mask(series, mask_terms):
    mwe = MWETokenizer(mask_terms)
    # punctuation_table = str.maketrans('', '', string.punctuation)

    masked = series.copy()
    ground_truth = pd.Series(index=series.index)

    for idx, s in masked.items():
        is_replaced = False
        gt = []
        # convert to lower case
        s = s.lower().replace('<br />', '')
        # nltk tokenize
        tokens = mwe.tokenize(word_tokenize(s))
        # remove punctuation from each token
        # tokens = [w.translate(punctuation_table) for w in tokens]
        for j in range(len(tokens)):
            if tokens[j] in mask_terms:
                gt.append(tokens[j])
                tokens[j] = MASK
                is_replaced = True
        masked[idx] = TreebankWordDetokenizer().detokenize(tokens) if is_replaced else pd.NA
        ground_truth[idx] = ' '.join(gt) if is_replaced else pd.NA
    return masked, ground_truth


# def preprocess(series, get_stats=True):
#     remove_words = {'br', 'nt'}
#     stop_words = set(stopwords.words('english'))
#     stop_words.update(remove_words)
#
#     multi_word_terms = set()
#     all_terms = set()
#     for key, categories in STEREOSET_TERMS.items():
#         for c in categories:
#             for s in c:
#                 split = s.split('_')
#                 all_terms.add(s)
#                 if len(split) > 1: multi_word_terms.add(s)
#     print(f'Multi-word bias terms: {multi_word_terms}')
#     mwe = MWETokenizer(multi_word_terms)
#
#     punctuation_table = str.maketrans('', '', string.punctuation)
#     tokenized = series.copy()
#
#     stats_df = pd.DataFrame(index=all_terms, columns=['occurrence'])
#     stats_df.fillna(0, inplace=True)
#     max_len = 0
#     total_len = 0
#     for i, s in series.items():
#         # convert to lower case
#         s = s.lower()
#         # nltk tokenize
#         tokens = mwe.tokenize(word_tokenize(s))
#         # remove punctuation from each token
#         tokens = [w.translate(punctuation_table) for w in tokens]
#         # remove stopwords
#         tokens = [w for w in tokens if w not in stop_words]
#         # remove hanging 's' and 'a'
#         tokens = [w for w in tokens if len(w) > 1]
#         # remove tokens with numbers in them
#         tokens = [w for w in tokens if w.isalpha()]
#         # # append starting and ending, store as string
#         # tokens = 'startseq ' + ' '.join(tokens) + ' endseq'
#         # get max length
#         if len(tokens) > max_len: max_len = len(tokens)
#         total_len += len(tokens)
#         # add to new pandas series
#         tokenized[i] = ' '.join(tokens)
#
#         if get_stats:
#             for m in all_terms:
#                 if m in tokens:
#                     stats_df.at[m, 'occurrence'] += 1
#             stats_df['probability'] = stats_df['occurrence']/float(total_len)
#     print(f'Max length of tokens: {max_len}')
#     if get_stats:
#         with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#             print(f'Statistics: {stats_df}')
#     return tokenized


if __name__ == '__main__':
    get_imdb_data()
