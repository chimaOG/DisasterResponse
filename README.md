Analyze and classify aid request emssages sent during disasters.

##Libraries Used##
-Flask==1.1.2
-gunicorn==20.0.4 (if you plan to deploy your app to Heroku)
-Jinja2==2.11.1
-joblib==0.14.1
-nltk==3.4.5
-numpy==1.18.2
-pandas==1.0.3
-plotly==4.6.0
-pylint==2.4.4
-python-dateutil==2.8.1
-scikit-learn==0.22.2.post1
-scipy==1.4.1
-six==1.14.0
-sklearn==0.0
-SQLAlchemy==1.3.16


##Project Motivation##

In the days following a disaster, emergency agencies are hit with a lot of message 
and they have to spend valuable manpower sorting these messages and chanelling them
to the responsible emergency agency. 

This project seeks to develop a tool that can make this process easier using machine learning.

This project involves using a large number of messages to train a machine learning model that 
can classify new messages based on the type of request and the specific type of aid being requested.

For the sake of displaying the model's operation. A simple webapp has been designed that takes in 
messages from the user and classifies it based on the type of request and the type of help being 
requested.


##Data Used##

The data used in the project consist of two data sets which are names 'disaster_messages.csv' and 
'disaster_categories.csv' - they can both be found in the data folder of this repo.

The disaster_messages.csv contains a dataset of messages sent in during emergency situations while the 
disaster_categories.csv contains the categories which the messages fall into. 

The data contained in both these files was imported, merged into one dataset and cleaned using pandas before
being stored in a databse. This process was executed using the process_data.py file.
 
The clean data from the databse was then passed through a machine learning pipeline that carried out processes
like normalization, lemmatization and tokenizing before using the data to train a random forest classifier.
The model was optimized using gridsearch and was stored in a pickle file. The process of training the model was
carried out using the train_classifier.py file. 

The data used in this project was provided by Figure Eight.

##Running ETL Scripy
1) Clone the repository
2) cd into the data folder
3) Run the following command: 'python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db'

##Building the Classifier Model
1) Clone the repository
2) cd into the models folder

##Hosting the webapp locally 

To host the app on your local machine:
1) Clone this reository
2) cd into the DisasterMSG folder 
3) run the command 'python run.py'

Ensure that all the libraries specified above are installed to prevent errors.
