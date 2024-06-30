import os
from werkzeug.utils import secure_filename
import time
import pandas as pd
from flask import jsonify

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xlsm', 'xls'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def update_file_name(file_name):
    ind_ext = file_name.find('.')
    return file_name[:ind_ext] + '_' + str(int(time.time())) + file_name[ind_ext:]


def remove_file_ext(file):
    ind_ext = file.find('.')
    return file[:ind_ext]


def open_file(file_path):
    df = pd.DataFrame({})
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xls'):
        df = pd.read_excel(file_path, engine='openpyxl')
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path, engine='openpyxl')
    elif file_path.endswith('.xlsm'):
        df = pd.read_excel(file_path, engine='openpyxl')
    return df

def save_and_return_column(file, directory, db_collection, session, session_key):
    if file and allowed_file(file.filename):

        file_name = secure_filename(file.filename)
        # print('file_name=', file_name)
        new_file_name = update_file_name(file_name)
        file_path = os.path.join(directory, new_file_name)
        file.save(file_path)

        try:
            user_file = db_collection.insert_one({
                'file_path': file_path
            })
        except Exception as e:
            return jsonify({'success': False, 'message': 'Something went wrong.'})
        session[session_key] = str(user_file.inserted_id)

        # print('Session Data after saving:', dict(session))
        file.seek(0)
        df = pd.DataFrame({})
        try:
            if file_name.endswith('.csv'):
                # print('reading csv file')
                df = pd.read_csv(file)
                # print('csv file read')
            elif file_name.endswith('.xls'):
                df = pd.read_excel(file, engine='openpyxl')
            elif file_name.endswith('.xlsx'):
                df = pd.read_excel(file, engine='openpyxl')
            elif file_name.endswith('.xlsm'):
                df = pd.read_excel(file, engine='openpyxl')

            col_names = df.columns.tolist()
            return jsonify({'success': True, 'columns': col_names})
        except Exception as e:
            print(f'An error occurred: {str(e)}')
            return jsonify({'success': False, 'message': 'Something went wrong.'})

    else:
        return jsonify({'success': False, 'message': 'File format not supported.'})
