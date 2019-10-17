# import libraries
import pickle
import string
import unittest

import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
nltk.download('stopwords')
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report


def load_data(database_filepath):
    '''
    Load the database and return X, Y
    --
    Inputs:
        database_filepath: database path
    Outputs:
        X: dependent variables for ML model
        Y: independent variable for ML model
        category_names: the columns in Y
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('messages', engine)
    X = df.message
    Y = df.loc[:, 'related':'direct_report']
    category_names = df.columns[4:]
    
    return X, Y, category_names


def tokenize(text):
    '''
    Tokenize the input text and return a clean tokens after normalization, lemmatization, and removing stopwords
    --
    Inputs:
        text: text
    Outputs:
        clean_tokens: a clean list of tokens
    '''
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    clean_tokens = [w for w in clean_tokens if w not in stopwords.words("english")]

    return clean_tokens


def build_model():
    '''
    Build a pipeline including CountVectorizer, tf-idf, MultiOutput RandomForestClassifier Classifier, and GridSearchCV  
    --
    Outputs:
        cv: better parameters Machine Learning pipeline
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 4]
    } 

    cv = GridSearchCV(pipeline, parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluate the model with it accurancy score
    --
    Inputs:
        model: a trained model
        X_test: features of test set
        Y_test: target values of test set
        category_names: names of each columns
    '''

    y_pred = model.predict(X_test)
    print(classification_report(y_pred, Y_test.values, target_names = category_names))
    

def save_model(model, model_filepath):
    '''
    Save the model
    Inputs:
        model: a trained model
        model_filepath: model file path
    '''
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
