import warnings
warnings.simplefilter('ignore')

import os, sys
import gc
import re
from collections import Counter

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 300)
from tqdm.auto import tqdm
tqdm.pandas()
import itertools

from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import GaussianNB
import category_encoders as ce
from sklearn.svm import SVC

import lightgbm as lgb

train_logs = pd.read_csv('/kaggle/input/linking-writing-processes-to-writing-quality/train_logs.csv')
display(train_logs)
train_scores = pd.read_csv('/kaggle/input/linking-writing-processes-to-writing-quality/train_scores.csv')
display(train_scores)
test_logs = pd.read_csv('/kaggle/input/linking-writing-processes-to-writing-quality/test_logs.csv')
display(test_logs)

activities = ['Input', 'Remove/Cut', 'Nonproduction', 'Replace', 'Paste']
events = ['q', 'Space', 'Backspace', 'Shift', 'ArrowRight', 'Leftclick', 'ArrowLeft', '.', ',', 
          'ArrowDown', 'ArrowUp', 'Enter', 'CapsLock', "'", 'Delete', 'Unidentified']
text_changes = ['q', ' ', 'NoChange', '.', ',', '\n', "'", '"', '-', '?', ';', '=', '/', '\\', ':']
punctuations = ['"', '.', ',', "'", '-', ';', ':', '?', '!', '<', '>', '/', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+']

def item_counts(df, itemList, itemName):
    temp_df = df.groupby('id').agg({itemName: list}).reset_index()
    res = list()
    for list_of_items in tqdm(temp_df[itemName].values):
        item_counts = list(Counter(list_of_items).items())
        item_dictionary = dict()
        for item in itemList:
            item_dictionary[item] = 0
        for counts in item_counts:
            item, item_sum = counts[0], counts[1]
            if item in item_dictionary:
                item_dictionary[item] = item_sum
        res.append(item_dictionary)
    res = pd.DataFrame(res)
    cols = [f'{itemName}_{i}_count' for i in range(len(res.columns))]
    res.columns = cols
    return res

def punctuation_count(df):
    temp_df = df.groupby('id').agg({'down_event': list}).reset_index()
    res = list()
    for punList in tqdm(temp_df['down_event'].values):
        counter = 0
        pun_counts = list(Counter(punList).items())
        for count in pun_counts:
            pun, pun_sum = count[0], count[1]
            if pun in punctuations:
                counter += pun_sum
            res.append(counter)
    res = pd.DataFrame({'punctuation_count': res})
    return res

def get_input_words(df):
    temp_df = df[(~df['text_change'].str.contains('=>'))&(df['text_change'] != 'NoChange')].reset_index(drop=True)
    temp_df = temp_df.groupby('id').agg({'text_change': list}).reset_index()
    temp_df['text_change'] = temp_df['text_change'].apply(lambda x: ''.join(x))
    temp_df['text_change'] = temp_df['text_change'].apply(lambda x: re.findall(r'q+', x))
    temp_df['input_word_count'] = temp_df['text_change'].apply(len)
    temp_df['input_word_length_mean'] = temp_df['text_change'].apply(lambda x: np.mean([len(i) for i in x] if len(x) > 0 else 0))
    temp_df['input_word_length_max'] = temp_df['text_change'].apply(lambda x: np.max([len(i) for i in x] if len(x) > 0 else 0))
    temp_df['input_word_length_std'] = temp_df['text_change'].apply(lambda x: np.std([len(i) for i in x] if len(x) > 0 else 0))
    temp_df.drop(['text_change'], axis=1, inplace=True)
    return temp_df

def make_features(df):
    
    # id
    feats = pd.DataFrame({'id': df['id'].unique().tolist()})
    
    # time shift
    df['up_time_shift1'] = df.groupby('id')['up_time'].shift(1)
    df['action_time_gap'] = df['down_time'] - df['up_time_shift1']
    df.drop('up_time_shift1', axis=1, inplace=True)
    
    # cursor position shift
    df['cursor_position_shift1'] = df.groupby('id')['cursor_position'].shift(1)
    df['cursor_position_change'] = np.abs(df['cursor_position'] - df['cursor_position_shift1'])
    df.drop('cursor_position_shift1', axis=1, inplace=True)
    
    # word count shift
    df['word_count_shift1'] = df.groupby('id')['word_count'].shift(1)
    df['word_count_change'] = np.abs(df['word_count'] - df['word_count_shift1'])
    df.drop('word_count_shift1', axis=1, inplace=True)
    
    # stats feats
    for item in tqdm([
        ('event_id', ['max']),
        ('up_time', ['max']),
        ('action_time', ['sum', 'max', 'mean', 'std']),
        ('activity', ['nunique']),
        ('down_event', ['nunique']),
        ('up_event', ['nunique']),
        ('text_change', ['nunique']),
        ('cursor_position', ['nunique', 'max', 'mean']),
        ('word_count', ['nunique', 'max', 'mean']),
        ('action_time_gap', ['max', 'min', 'mean', 'std', 'sum']),
        ('cursor_position_change', ['max', 'mean', 'std', 'sum']),
        ('word_count_change', ['max', 'mean', 'std', 'sum'])
    ]):
        colname, methods = item[0], item[1]
        for method in methods:
            temp_df = df.groupby(['id']).agg({colname: method}).reset_index().rename(columns={colname: f'{colname}_{method}'})
            feats = feats.merge(temp_df, on='id', how='left')
    
    # counts
    temp_df = item_counts(df, activities, 'activity')
    feats = pd.concat([feats, temp_df], axis=1)
    temp_df = item_counts(df, events, 'down_event')
    feats = pd.concat([feats, temp_df], axis=1)
    temp_df = item_counts(df, events, 'up_event')
    feats = pd.concat([feats, temp_df], axis=1)
    temp_df = item_counts(df, text_changes, 'text_change')
    feats = pd.concat([feats, temp_df], axis=1)
    temp_df = punctuation_count(df)
    feats = pd.concat([feats, temp_df], axis=1)
    
    # input words
    temp_df = get_input_words(df)
    feats = pd.merge(feats, temp_df, on='id', how='left')
    
    # compare feats
    feats['word_time_ratio'] = feats['word_count_max'] / feats['up_time_max']
    feats['word_event_ratio'] = feats['word_count_max'] / feats['event_id_max']
    feats['event_time_ratio'] = feats['event_id_max']  / feats['up_time_max']
    feats['idle_time_ratio'] = feats['action_time_gap_sum'] / feats['up_time_max']
    
    return feats

train_feats = make_features(train_logs)
test_feats = make_features(test_logs)

train_feats = train_feats.merge(train_scores, on='id', how='left')
train_feats = train_feats.dropna()

X = train_feats.drop(["id","score"], axis=1)
y = train_feats["score"].to_numpy()

numeric_features = X.select_dtypes([np.number]).columns
categorical_features = X.select_dtypes(exclude=[np.number]).columns

from sklearn import preprocessing
y_convert = preprocessing.LabelEncoder()
y_transformed = y_train_convert.fit_transform(y)
print("y_transformed")
print(y_transformed)

X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, test_size=0.2, random_state=44)

models = {
    "Random Forest": RandomForestClassifier(n_estimators=50, max_features="sqrt", random_state=44)
}

encoders = {
    'OneHotEncoder': ce.one_hot.OneHotEncoder
}

try:
    model = pipe.fit(X_train, y_train)
    y_pred = model.predict(X_test)
except Exception as error:
   fail = True
   print(error)
   print("fail")

print(X_test.index.values)
print(y_pred)

submission = pd.DataFrame({'id': X_test.index.values, 'score' : np.squeeze(y_pred)})
