from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_session import Session
from pymongo import MongoClient
import os
from fileUtils import save_and_return_column
from trainAndSaveModel import train_and_save_model
from modelPrediction import model_prediction
import time


app = Flask(__name__)
CORS(app, supports_credentials=True)

# app.config['MONGO_URI'] = 'mongodb://localhost:27017/dl_cell_predict'
app.config["MONGODB_SETTINGS"] = {
    "db": "dl_cell_predict",
    "host": "localhost",
    "port": 27017,
    "alias": "default",
}

# mongo = PyMongo(app)
# session parameters
app.config['SESSION_TYPE'] = 'mongodb'
# app.config['SESSION_MONGODB'] = mongo.cx
app.config['SESSION_MONGODB'] = MongoClient(host=app.config["MONGODB_SETTINGS"]["host"],
                                            port=app.config["MONGODB_SETTINGS"]["port"])
app.config['SESSION_MONGODB_DB'] = app.config["MONGODB_SETTINGS"]["db"]
app.config['SESSION_MONGODB_COLLECT'] = 'sessions'
app.config['SESSION_PERMANENT'] = False  # Make the session non-permanent
app.config['SESSION_USE_SIGNER'] = True  # Encrypt the session data
# app.config['SECRET_KEY'] = os.urandom(24)  # Use for cryptographic operations
app.config['SECRET_KEY'] = 'secret_key'

# Initialize the session
Session(app)

# Initialize MongoDB client for custom data storage
mongo_client = MongoClient(
    host=app.config["MONGODB_SETTINGS"]["host"],
    port=app.config["MONGODB_SETTINGS"]["port"]
)
db = mongo_client[app.config["MONGODB_SETTINGS"]["db"]]
user_files_collection = db['user_files']

# Other application parameters
app.config['FILE_UPLOAD_FOLDER'] = "user_files"
app.config['MODEL_SAVE_FOLDER'] = "trained_model"
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
    time.sleep(10)
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
