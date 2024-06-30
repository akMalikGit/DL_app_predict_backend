from flask import jsonify
from bson import ObjectId
import os
from trainModel import train_model


def train_and_save_model(selected_columns, target_column, learning_rate, is_categorical, model_directory, db_collection,
                         session, session_key):
    if not selected_columns or not target_column or not learning_rate:
        return jsonify({'success': False, 'message': 'Missing parameters.'})

    file_id = session.get(session_key)
    # print('session get file_id', file_id)
    if not file_id:
        return jsonify({'success': False, 'message': 'No file uploaded.'})

    # Fetch file info from the database
    file_path = db_collection.find_one({'_id': ObjectId(file_id)})['file_path']
    # print('user_file', user_file)
    if not file_path:
        # print("file not found for user")
        return jsonify({'success': False, 'message': 'File not found'})

    # Train the model
    try:
        model, model_name, target_props, d_str_selected_uq = train_model(file_path,
                                                                         selected_columns,
                                                                         target_column,
                                                                         is_categorical,
                                                                         learning_rate)
    except:
        return jsonify({'success': False, 'message': 'Something went wrong.'})

    model_path = os.path.join(model_directory, model_name)
    model.save(model_path)
    # Save model path to the database
    try:
        db_collection.update_one({'_id': ObjectId(file_id)}, {
            '$set': {'model_path': model_path, 'model_name': model_name, 'target_props': target_props}})
    except:
        return jsonify({'success': False, 'message': 'Something went wrong.'})

    return jsonify({'success': True, 'message': 'Model trained successfully', 'd_str_selected_uq': d_str_selected_uq})
