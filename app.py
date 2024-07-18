from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_session import Session
from pymongo import MongoClient
import os
from fileUtils import save_and_return_column
from trainAndSaveModel import train_and_save_model
from modelPrediction import model_prediction
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app, supports_credentials=True)

# Load environment variables from .env file
load_dotenv()

# Load environment variables from .env file
load_dotenv()

# Fetch credentials from environment variables
MONGO_DB = os.getenv("MONGO_DB", "Database info not found")
MONGO_HOST = os.getenv("MONGO_HOST", "localhost")
MONGO_PORT = int(os.getenv("MONGO_PORT", 27017))
MONGO_USER = os.getenv("MONGO_USER", "user_not_found")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD", "password_not_found")
SECRET_KEY = os.getenv("SECRET_KEY", "default_secret_key")
AUTH_MECH = os.getenv("AUTH_MECH", "SCRAM-SHA-256")
DB_SESSIONS_COLL_NAME = os.getenv("DB_SESSIONS_COLL_NAME", "collection_not_defined")
DB_FILE_COLL_NAME = os.getenv("DB_FILE_COLL_NAME", "collection_not_defined")
FILE_UPLOAD_FOLDER = os.getenv("FILE_UPLOAD_FOLDER", "collection_not_defined")
MODEL_SAVE_FOLDER = os.getenv("MODEL_SAVE_FOLDER", "collection_not_defined")

app.config["MONGODB_SETTINGS"] = {
    "db": MONGO_DB,
    "host": MONGO_HOST,
    "port": MONGO_PORT,
    "alias": "default",
}

mongo_client = MongoClient(
    host=app.config["MONGODB_SETTINGS"]["host"],
    port=app.config["MONGODB_SETTINGS"]["port"]
    # un-comment below parameters authentication is enabled in database
    # username=MONGO_USER,
    # password=MONGO_PASSWORD,
    # authSource=MONGO_DB,
    # authMechanism=AUTH_MECH
)

# session parameters
app.config['SESSION_TYPE'] = 'mongodb'
app.config['SESSION_MONGODB'] = mongo_client
app.config['SESSION_MONGODB_DB'] = app.config["MONGODB_SETTINGS"]["db"]
app.config['SESSION_MONGODB_COLLECT'] = DB_SESSIONS_COLL_NAME
app.config['SESSION_PERMANENT'] = False  # Make the session non-permanent
app.config['SESSION_USE_SIGNER'] = True  # Encrypt the session data
app.config['SECRET_KEY'] = SECRET_KEY

# Initialize the session
Session(app)

# Initialize MongoDB client for custom data storage
db = mongo_client[app.config["MONGODB_SETTINGS"]["db"]]
user_files_collection = db[DB_FILE_COLL_NAME]

# Other application parameters
app.config['FILE_UPLOAD_FOLDER'] = FILE_UPLOAD_FOLDER
app.config['MODEL_SAVE_FOLDER'] = MODEL_SAVE_FOLDER
app.config['SESSION_KEY'] = 'file_id'

if not os.path.exists(app.config['FILE_UPLOAD_FOLDER']):
    os.makedirs(app.config['FILE_UPLOAD_FOLDER'])
if not os.path.exists(app.config['MODEL_SAVE_FOLDER']):
    os.makedirs(app.config['MODEL_SAVE_FOLDER'])


@app.route('/')
def hello():
    return 'Hello, World! This application is based on deep learning.'


@app.route('/upload', methods=['POST'])
def user_file_upload():
    # print('files:', request.files)
    # print('uploaded_file:', request.files['file'])
    # print('file_name:', request.files['file'].filename)
    if len(request.files) == 0:
        return jsonify({'success': False, 'message': 'No Selected File.'})

    uploaded_file = request.files['file']
    return save_and_return_column(file=uploaded_file,
                                  directory=app.config['FILE_UPLOAD_FOLDER'],
                                  db_collection=user_files_collection,
                                  session=session,
                                  session_key=app.config['SESSION_KEY'])


@app.route('/train', methods=['POST'])
def user_param_modal_train():
    data = request.json
    selected_columns = data.get('selectedColumns')
    target_column = data.get('targetColumn')
    learning_rate = float(data.get('learningRate'))
    is_categorical = data.get('isCategorical')
    # print(is_categorical)
    # print('selected_col=', selected_columns)
    # print('target_col=', target_column)
    # print('learning_rate=', learning_rate)

    return train_and_save_model(selected_columns,
                                target_column,
                                learning_rate,
                                is_categorical,
                                model_directory=app.config['MODEL_SAVE_FOLDER'],
                                db_collection=user_files_collection,
                                session=session,
                                session_key=app.config['SESSION_KEY'])


@app.route('/test', methods=['POST'])
def user_param_modal_predict():
    data = request.json
    return model_prediction(data,
                            db_collection=user_files_collection,
                            session=session,
                            session_key=app.config['SESSION_KEY'])


if __name__ == '__main__':
    app.run(debug=True)
