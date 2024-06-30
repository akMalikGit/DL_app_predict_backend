from flask import jsonify
from bson import ObjectId
import tensorflow as tf
import numpy as np


def preprocess_input(input_dict, model):
    processed_dict = {}
    # print('pre_processing')
    for layer in model.layers:
        layer_config = layer.get_config()
        layer_name = layer_config['name']
        # print('model_layer=', layer_name)
        if layer_name in input_dict:
            # print('\tmodel_lyr found in user data')
            # Convert input value to the correct dtype
            if layer_config['dtype'] == 'float32':
                # print('\ttype = float')
                processed_dict[layer_name] = np.array([float(input_dict[layer_name])], dtype=np.float32)
            else:
                # Add more dtype handling if needed
                # print('\ttype = string')
                processed_dict[layer_name] = np.array([input_dict[layer_name]], dtype='O')
    # print('processed_dict', processed_dict)
    return processed_dict


# Create input tensors from the preprocessed dictionary
def create_input_tensors(processed_dict, model):
    input_tensors = []
    for layer in model.layers:
        layer_name = layer.get_config()['name']
        if layer_name in processed_dict:
            input_tensors.append(processed_dict[layer_name])
    # print('input_tensors=', input_tensors)
    return input_tensors


def predict_output(input_dict, model, target_props):
    # Preprocess the input dictionary
    processed_dict = preprocess_input(input_dict, model)

    # Create input tensors
    input_tensors = create_input_tensors(processed_dict, model)

    # Predict the output
    # Assuming your model accepts multiple inputs, wrap them in a list
    prediction = model.predict(input_tensors)

    # print("Prediction:", prediction)
    if target_props['dtype'] == "<dtype: 'float32'>":
        prediction = (prediction * target_props['std']) + target_props['mean']
        predicted_value = float(prediction[0][0])
        formatted_prediction = str(f"{predicted_value:.4f}")
        # print("Final Prediction=", formatted_prediction)
    else:
        vocab = target_props['vocab']
        lookup_layer = tf.keras.layers.StringLookup(vocabulary=vocab)
        inverse_lookup_layer = tf.keras.layers.StringLookup(invert=True, vocabulary=vocab)
        prediction = np.argmax(prediction, axis=1)
        prediction = inverse_lookup_layer(prediction)
        formatted_prediction= str(prediction.numpy()[0].decode('utf-8'))
        # print("Predicted class:", prediction.numpy())
    return formatted_prediction


def model_prediction(data, db_collection, session, session_key):
    file_id = session.get(session_key)
    if not file_id:
        return jsonify({'success': False, 'message': 'Not a valid session.'})

    # Fetch file info from the database
    user_file_details = db_collection.find_one({'_id': ObjectId(file_id)})
    if not user_file_details:
        return jsonify({'success': False, 'message': 'No data found'})
    model_path = user_file_details['model_path']
    # print('user_file', model_path)
    model = tf.keras.models.load_model(model_path)

    target_props = user_file_details['target_props']
    try:
        prediction = predict_output(data, model, target_props)
    except:
        return jsonify({'success': False, 'message': "Something went wrong."})

    return jsonify({'success': True, 'message': str(prediction)})
