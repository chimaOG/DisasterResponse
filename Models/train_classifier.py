# import libraries
import pandas as pd
import numpy as np
from sqlalchemy.engine import create_engine
import re
import pickle


from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report
from sklearn.model_selection import GridSearchCV

from scipy.stats import hmean
from scipy.stats.mstats import gmean


import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM DisasterMessages', con = engine)
    
    #Split data into x and y 
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    
    return X, Y, category_names

def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    tokens = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = [] 
    for toks in tokens:
        clean_tokens.append(lemmatizer.lemmatize(toks).lower().strip())
    
    return clean_tokens


def build_model():
    multi = MultiOutputClassifier(RandomForestClassifier(n_estimators = 10))


    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer = tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', multi)
            ])

    #parameters = {'clf__estimator__n_estimators': [50, 100],
     #           'clf__estimator__min_samples_split': [2, 3, 4],
      #            'clf__estimator__criterion': ['entropy', 'gini']
       #          }
    #cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return pipeline

def multioutput_fscore(y_true,y_pred,beta=1):
    """
    MultiOutput Fscore
    
    This is a performance metric of my own creation.
    It is a sort of geometric mean of the fbeta_score, computed on each label.
    
    It is compatible with multi-label and multi-class problems.
    It features some peculiarities (geometric mean, 100% removal...) to exclude
    trivial solutions and deliberatly under-estimate a stangd fbeta_score average.
    The aim is avoiding issues when dealing with multi-class/multi-label imbalanced cases.
    
    It can be used as scorer for GridSearchCV:
        scorer = make_scorer(multioutput_fscore,beta=1)
        
    Arguments:
        y_true -> labels
        y_prod -> predictions
        beta -> beta value of fscore metric
    
    Output:
        f1score -> customized fscore
    """
    score_list = []
    if isinstance(y_pred, pd.DataFrame) == True:
        y_pred = y_pred.values
    if isinstance(y_true, pd.DataFrame) == True:
        y_true = y_true.values
    for column in range(0,y_true.shape[1]):
        score = fbeta_score(y_true[:,column],y_pred[:,column],beta,average='weighted')
        score_list.append(score)
    f1score_numpy = np.asarray(score_list)
    f1score_numpy = f1score_numpy[f1score_numpy<1]
    f1score = gmean(f1score_numpy)
    return  f1score


def evaluate_model(model, X_test, Y_test, category_names):
    
     """
     Evaluate Model function
    
     This function applies ML pipeline to a test set and prints out
     model performance (accuracy and f1score)
    
     Arguments:
        model -> Scikit ML Pipeline
        X_test -> test features
        Y_test -> test labels
        category_names -> label names (multi-output)
     """
     Y_pred = model.predict(X_test)
    
     multi_f1 = multioutput_fscore(Y_test,Y_pred, beta = 1)
     overall_accuracy = (Y_pred == Y_test).mean().mean()

     print('Average overall accuracy {0:.2f}% \n'.format(overall_accuracy*100))
     print('F1 score (custom definition) {0:.2f}%\n'.format(multi_f1*100))
     pass


def save_model(model, model_filepath):
    """
    Save Model function
    
    This function saves trained model as Pickle file, to be loaded later.
    
    Arguments:
        model -> GridSearchCV or Scikit Pipelin object
        model_filepath -> destination path to save .pkl file
    
    """
    pickle.dump(model, open(model_filepath, 'wb'))
    pass


def main():
    database_filepath, model_filepath = '..\Data\DisasterResponse.db', 'classifier.pkl'
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



if __name__ == '__main__':
    main()