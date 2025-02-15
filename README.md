# Machine Learning Model Training and Prediction Application

This application allows users to upload Excel or CSV files to train a machine learning model and make predictions based on the provided data. The backend is built using Flask, and it uses MongoDB to store session details and uploaded files.

## Features

- Upload Excel or CSV files for training
- Train a machine learning model from the uploaded data
- Make predictions using the trained model
- Store session information and uploaded files in MongoDB
- Provides endpoints for training and prediction

## Prerequisites

- Python 3.9.13
- MongoDB

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/akMalikGit/DL_app_predict_backend.git
cd your-repository/backend
```
## Install Python Dependencies
Install the required Python packages using pip:
```
pip install -r requirements.txt
```

## Install MongoDB
### For Windows
- Download MongoDB from the official website (https://www.mongodb.com/try/download/community).
- Follow the installation instructions.
- Start the MongoDB server by running mongod in the command prompt.
### For Linux
Follow the instructions on the official MongoDB installation guide (https://www.mongodb.com/docs/manual/administration/install-on-linux/).
```
sudo service mongod start
```

## Create Database and Collections
- Connect to the MongoDB server using the MongoDB shell or a GUI tool like MongoDB Compass.
- Create a database named **ml_app** or any other name.
- Create the necessary collections for your application, e.g., **sessions, files**.

## Configure the Application
Create and update the parameter values in .env file.
```text
MONGO_DB="database"
MONGO_HOST="localhost"
MONGO_PORT="27017"
MONGO_USER="user_name"
MONGO_PASSWORD="password"
SECRET_KEY = "default_secret_key"
AUTH_MECH = "SCRAM-SHA-256"
DB_SESSIONS_COLL_NAME = "sessions_collection_name"
DB_FILE_COLL_NAME = "user_files_collection_name"
FILE_UPLOAD_FOLDER = "user_files_directory_path"
MODEL_SAVE_FOLDER = "trained_model_directory_path"
```
Update the configuration settings in your Flask application as needed. This might include MongoDB connection details, file paths, and other settings.


**Update app.py**

If you have enabled authorization on the database please enable the parameters of *MongoClient* method
(Update app.py line no 43)

## Start the Application
Run the Flask application:
```bash
export FLASK_APP=app.py
flask run
```
or for windows:
```cmd
set FLASK_APP=app.py
flask run
```
or run using command:
```
python app.py
```
The application should now be running on http://127.0.0.1:5000.

