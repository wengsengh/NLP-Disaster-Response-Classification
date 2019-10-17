# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
    '''
    Load dataframes for messages and categories data.
  
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')


    return df


def clean_data(df):
    '''    
    Load a dataframe and clean it 

    '''

    # Create categories dataframe that consists 36 individual category columns
    categories = df.categories.str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = df.categories.str.split(';',expand=True).loc[0]

    # use this row to extract a list of new column names for categories.
    col_names = [categories[:-2] for categories in row ]

    # Rename the col_names to categories
    categories.columns = col_names

    # Set each value to be the last character of the string and convert into integer
    for col in categories.columns:
        categories[col] = categories[col].str[-1]
        #convert to integer
        categories[col] = categories[col].astype(int)

    # Concatenate the original dataframe with the new dataframe
    df = pd.concat([df.drop('categories', axis=1), categories], axis=1)

    # Remove 'related = 2'
    df = df[(df['related'] == 1) | (df['related'] == 0)]

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    '''
    Load a dataframe and save it to a database

    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('messages', engine, index=False, chunksize=1000, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
