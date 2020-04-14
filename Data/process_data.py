#Importing Libraries 
import sys
import pandas as pd
from  sqlalchemy.engine import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Inputs:
        messages_filepath - File path for the csv file with the messages
        categories_filepath - File path for the csv file with the message categories
        
    Returns:
        A single data frame that contains the data in both files.
    '''
    
    #Import the files
    categories = pd.read_csv(categories_filepath)
    messages = pd.read_csv(messages_filepath)
    
    #Merge both files using the id column as key
    df = messages.merge(categories, on = 'id', how = 'outer')
    return df


def clean_data(df):
    '''    
    Inputs:
        df - a pandas dataframe
        
    Returns:
        A dataframe that is the cleaned and feature engineered version of the original one.
    '''
    #Split the categories column into multiple columns where each column is a category
    categories = df['categories'].str.split(';', expand = True)
    
    #Name the newly created columns
    
    # select the first row of the categories dataframe
    row = categories.iloc[0, :]
    #Extract the category names from the row
    category_colnames = []
    row.apply(lambda x: category_colnames.append(x[:-2]))
    
    #replace the column names with teh newlt extracted category names
    categories.columns = category_colnames
    
    #convert the data in the category dataframe to numerical data.
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x : x[-1])
    
    # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    #Drop the original categories dataframe  and append the cleaned categories dataframe
    df.drop(['categories'], axis = 1, inplace = True)
    df = pd.concat([df, categories], axis = 1)
    
    #drop any duplicates in the concatenated dataframe
    df.drop_duplicates(inplace = True)
    
    return df



def save_data(df, database_filename):
    '''    
    Inputs:
        df - a pandas dataframe
        database_filename - the name of the database file.
    '''
    
    #Create the connection engine and push the data to the DB
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterMessages', engine, index=False)


def main():
    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        #Cleaning the Data
        print('Cleaning data...')
        df = clean_data(df)
        
        #Save the cleaned data to the DB
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