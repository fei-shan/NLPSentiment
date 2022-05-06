import os

import pandas as pd
from itertools import chain

from config import (
    STEREOSET_TERMS,
    LOCAL_DATA_DIR,
    PREDICT_IMDB_CSV
)

# PREDICT_IMDB_CSV = os.path.join(LOCAL_DATA_DIR, 'predict_imdb_bert.csv')
# PREDICT_IMDB_CSV = os.path.join(LOCAL_DATA_DIR, 'predict_imdb_bert_large.csv')


def analyze():
    df = pd.read_csv(PREDICT_IMDB_CSV, index_col=0)
    # df.rename(columns={df.columns[0]: 'original_index'}, inplace=True)
    # df['masked'] = df['masked'].str.replace('" ', '"')
    # df.to_csv(PREDICT_IMDB_CSV)
    print(df)

    df['ground_truth'] = df['ground_truth'].str.split(' ')
    df['prediction'] = df['prediction'].str.split(' ')
    # Only get the predicted masks
    for idx, row in df.iterrows():
        if type(row['prediction']) is list:
            df.at[idx, 'ground_truth'] = row['ground_truth'][:len(row['prediction'])]
        else:
            df.drop([idx], inplace=True)
    result_df = df.explode(['ground_truth', 'prediction'], ignore_index=True)
    # result_df = df.apply(pd.Series.explode, ignore_index=True)
    print(result_df)
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(result_df[['original_index', 'ground_truth', 'prediction']].iloc[:100])

    # Total prediction
    # NUM_PREDICTIONS = 272769
    gender = STEREOSET_TERMS['gender']
    male_terms = gender[0]
    female_terms = gender[1]

    gender_terms = list(chain.from_iterable(gender))
    race_terms = list(chain.from_iterable(STEREOSET_TERMS['race']))
    profession_terms = list(chain.from_iterable(STEREOSET_TERMS['profession']))
    religion_terms = list(chain.from_iterable(STEREOSET_TERMS['religion']))

    correct_prediction = result_df[result_df['ground_truth'] == result_df['prediction']]
    male_prediction = result_df[result_df['prediction'].isin(male_terms)]
    female_prediction = result_df[result_df['prediction'].isin(female_terms)]
    male_ground_truth = result_df[result_df['ground_truth'].isin(male_terms)]
    female_ground_truth = result_df[result_df['ground_truth'].isin(female_terms)]
    print(f'>>>correct_prediction: '
          f'{len(correct_prediction) / float(len(result_df))}'
          f'\nmale {len(male_prediction[male_prediction["prediction"]==male_prediction["ground_truth"]]) / float(len(male_ground_truth))}'
          f'\nfemale {len(female_prediction[female_prediction["prediction"]==female_prediction["ground_truth"]]) / float(len(female_ground_truth))}')

    print(f'>>>male_prediction: '
          f'{len(male_prediction) / float(len(result_df))}')
    print(f'>>>female_prediction: '
          f'{len(female_prediction) / float(len(result_df))}')

    print(f'>>>male_ground_truth: '
          f'{len(male_ground_truth) / float(len(result_df))}')
    print(f'>>>female_ground_truth: '
          f'{len(female_ground_truth) / float(len(result_df))}')

    non_gender_ground_truth = result_df[~result_df['ground_truth'].isin(male_terms)
                                        & ~result_df['ground_truth'].isin(female_terms)]
    non_gender_prediction = result_df[~result_df['prediction'].isin(male_terms)
                                      & ~result_df['prediction'].isin(female_terms)]
    print(f'>>>non_gender_prediction: '
          f'{len(non_gender_prediction) / float(len(result_df))}')

    print(f'>>>non_gender_ground_truth: '
          f'{len(non_gender_ground_truth) / float(len(result_df))}')

    male_genderized = non_gender_ground_truth[non_gender_ground_truth["prediction"].isin(male_terms)]
    female_genderized = non_gender_ground_truth[non_gender_ground_truth["prediction"].isin(female_terms)]
    print(f'>>>genderized_prediction: '
          f'\nmale {len(male_genderized) / float(len(non_gender_ground_truth))}'
          f'\nfemale {len(female_genderized) / float(len(non_gender_ground_truth))}')

    male_ground_truth = result_df[result_df['ground_truth'].isin(male_terms)]
    male_degenderized = male_ground_truth[~male_ground_truth["prediction"].isin(gender_terms)]
    female_ground_truth = result_df[result_df['ground_truth'].isin(female_terms)]
    female_degenderized = female_ground_truth[~female_ground_truth["prediction"].isin(gender_terms)]
    print(f'>>>degenderized_prediction: '
          f'\nmale {len(male_degenderized) / float(len(male_ground_truth))}'
          f'\nfemale {len(female_degenderized) / float(len(female_ground_truth))}')

    positive_prediction = result_df[result_df['label'] > 0]
    male_positive_prediction = positive_prediction[positive_prediction['prediction'].isin(male_terms)]
    female_positive_prediction = positive_prediction[positive_prediction['prediction'].isin(female_terms)]
    negative_prediction = result_df[result_df['label'] < 1]
    male_negative_prediction = negative_prediction[negative_prediction['prediction'].isin(male_terms)]
    female_negative_prediction = negative_prediction[negative_prediction['prediction'].isin(female_terms)]
    # print(f'>>>p(m): {len(result_df[result_df["ground_truth"].isin(gender_terms[0])]) / float(len(result_df))}')
    # print(f'>>>p(f): {len(result_df[result_df["ground_truth"].isin(gender_terms[1])]) / float(len(result_df))}')
    print(f'>>>p(m|p): { len(male_positive_prediction)/float(len(positive_prediction))}')
    print(f'>>>p(m|n): {len(male_negative_prediction) / float(len(negative_prediction))}')
    print(f'>>>p(f|p): {len(female_positive_prediction) / float(len(positive_prediction))}')
    print(f'>>>p(f|n): {len(female_negative_prediction) / float(len(negative_prediction))}')

    gender_ground_truth = result_df[result_df['ground_truth'].isin(gender_terms)]
    race_ground_truth = result_df[result_df['ground_truth'].isin(race_terms)]
    religion_ground_truth = result_df[result_df['ground_truth'].isin(religion_terms)]
    profession_ground_truth = result_df[result_df['ground_truth'].isin(profession_terms)]

    gender_prediction = result_df[result_df['prediction'].isin(gender_terms)]
    race_prediction = result_df[result_df['prediction'].isin(race_terms)]
    religion_prediction = result_df[result_df['prediction'].isin(religion_terms)]
    profession_prediction = result_df[result_df['prediction'].isin(profession_terms)]

    print(f'>>>P(g in gender): {len(gender_ground_truth) / float(len(result_df))}')
    print(f'>>>P(g in race): {len(race_ground_truth) / float(len(result_df))}')
    print(f'>>>P(g in religion): {len(religion_ground_truth) / float(len(result_df))}')
    print(f'>>>P(g in profession): {len(profession_ground_truth) / float(len(result_df))}')
    print(f'>>>P(p in gender): {len(gender_prediction) / float(len(result_df))}')
    print(f'>>>P(p in race): {len(race_prediction) / float(len(result_df))}')
    print(f'>>>P(p in religion): {len(religion_prediction) / float(len(result_df))}')
    print(f'>>>P(p in profession): {len(profession_prediction) / float(len(result_df))}')

    print(male_genderized.groupby(['ground_truth'])['ground_truth'].count().reset_index(name='Count').sort_values(
        ['Count'], ascending=False).iloc[:10])
    male_genderized.to_csv(os.path.join(LOCAL_DATA_DIR, 'male_genderized_bert.csv'))
    print(female_genderized.groupby(['ground_truth'])['ground_truth'].count().reset_index(name='Count').sort_values(
        ['Count'], ascending=False).iloc[:10])
    female_genderized.to_csv(os.path.join(LOCAL_DATA_DIR, 'female_genderized_bert.csv'))
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(male_genderized[['ground_truth', 'prediction']].iloc[:100])
    #     print(female_genderized[['ground_truth', 'prediction']].iloc[:100])

    # degenderize_prediction = result_df[result_df['ground_truth'].isin(gender_terms[0])
    #                                    | result_df['ground_truth'].isin(gender_terms[1])]
    # degenderize_prediction = degenderize_prediction[~degenderize_prediction['prediction'].isin(gender_terms[0])
    #                                                 & ~degenderize_prediction['prediction'].isin(gender_terms[1])]
    # false_male_prediction = result_df[~result_df['ground_truth'].isin(gender_terms[0])
    #                                   & result_df['prediction'].isin(gender_terms[0])]
    # false_female_prediction = result_df[~result_df['ground_truth'].isin(gender_terms[1])
    #                                     & result_df['prediction'].isin(gender_terms[1])]
    # print(f'>>>correct_prediction: {len(correct_prediction)/float(len(result_df))}')
    # print(f'>>>degenderize_prediction: {degenderize_prediction}')
    # print(f'>>>false_male_prediction: {false_male_prediction}')
    # print(f'>>>positive false_male_prediction: {false_male_prediction[false_male_prediction["label"] > 0]}')
    # print(f'>>>false_female_prediction: {false_female_prediction}')
    # print(f'>>>positive false_female_prediction: {false_female_prediction[false_female_prediction["label"] > 0]}')
    #
    # positive_prediction = result_df[result_df['label'] > 0]
    # male_positive_prediction = positive_prediction[positive_prediction['prediction'].isin(gender_terms[0])]
    # female_positive_prediction = positive_prediction[positive_prediction['prediction'].isin(gender_terms[1])]
    # negative_prediction = result_df[result_df['label'] < 1]
    # male_negative_prediction = negative_prediction[negative_prediction['prediction'].isin(gender_terms[0])]
    # female_negative_prediction = negative_prediction[negative_prediction['prediction'].isin(gender_terms[1])]
    # print(f'>>>p(m): {len(result_df[result_df["ground_truth"].isin(gender_terms[0])]) / float(len(result_df))}')
    # print(f'>>>p(f): {len(result_df[result_df["ground_truth"].isin(gender_terms[1])]) / float(len(result_df))}')
    # print(f'>>>p(m): {len(result_df[result_df["prediction"].isin(gender_terms[0])])/float(len(result_df))}')
    # print(f'>>>p(f): {len(result_df[result_df["prediction"].isin(gender_terms[1])])/float(len(result_df))}')
    # print(f'>>>p(m|p): { len(male_positive_prediction)/float(len(positive_prediction))}')
    # print(f'>>>p(m|n): {len(male_negative_prediction) / float(len(negative_prediction))}')
    # print(f'>>>p(f|p): {len(female_positive_prediction) / float(len(positive_prediction))}')
    # print(f'>>>p(f|n): {len(female_negative_prediction) / float(len(negative_prediction))}')
    #
    # male_ground_truth = result_df[result_df['ground_truth'].isin(gender_terms[0])]
    # female_ground_truth = result_df[result_df['ground_truth'].isin(gender_terms[1])]
    # print(f'>>>p(f|m): {len(male_ground_truth[male_ground_truth["prediction"].isin(gender_terms[1])])/float(len(male_ground_truth))}')
    # print(f'>>>p(m|f): {len(female_ground_truth[female_ground_truth["prediction"].isin(gender_terms[0])])/float(len(female_ground_truth))}')


if __name__ == '__main__':
    analyze()