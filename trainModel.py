import tensorflow as tf
import os
import numpy as np
from fileUtils import remove_file_ext, open_file
from modelUtils import draw_history_graph, custom_callbacks



def get_out_dtype(train_out, is_categorical):
    if is_categorical:
        return "string"

    return tf.string if train_out.dtype == object else tf.float32


def get_computation_model(pre_processing_model, input_layer, learning_rate, out_layer_shape, out_dtype):
    model_body = tf.keras.Sequential([
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
    ])
    # print('model_body_added')
    # print('train_out=', train_out)
    if out_dtype == tf.float32:
        model_body.add(tf.keras.layers.Dense(out_layer_shape))
    else:
        model_body.add(tf.keras.layers.Dense(out_layer_shape, activation='softmax'))

    pre_processed_inputs = pre_processing_model(input_layer)
    result = model_body(pre_processed_inputs)
    model = tf.keras.Model(input_layer, result)
    # print('final_modal =')
    # print(model.summary())

    if out_dtype == tf.float32:
        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      metrics=[tf.keras.metrics.MeanAbsoluteError()])
    else:
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      metrics=['accuracy'])
    return model


def get_input_layer(train_in):
    input_layer = {}
    for name, column in train_in.items():
        dtype = column.dtype
        if dtype == object:
            dtype = tf.string
        else:
            dtype = tf.float32
        input_layer[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)
    return input_layer


def train_model(file_path, selected_columns, target_column, is_categorical, learning_rate):
    max_categorical = 15
    df = open_file(file_path)
    d_str_selected_uq = {}

    # Split the data into inputs and target
    train_in = df[selected_columns]
    train_out = df[target_column]
    if is_categorical and len(np.unique(train_out)) < max_categorical:
        train_out = df[target_column].astype(str)
    else:
        is_categorical = False

    # print('train_model')
    # print(is_categorical)
    out_dtype = get_out_dtype(train_out, is_categorical)
    # print('updated OUT_DTYPE', out_dtype)

    pre_processed_in_lyr = []
    input_layer = get_input_layer(train_in)
    numeric_inputs = {name: input_tensor for name, input_tensor in input_layer.items() if
                      input_tensor.dtype == tf.float32}
    # print('numeric_inputs=', numeric_inputs)
    if len(numeric_inputs) > 0:
        concat_lyr = tf.keras.layers.Concatenate()(list(numeric_inputs.values()))
        norm_lyr = tf.keras.layers.Normalization()

        train_in_np = train_in[numeric_inputs.keys()].to_numpy()
        norm_lyr.adapt(train_in_np)
        norm_numeric_lyr = norm_lyr(concat_lyr)

        pre_processed_in_lyr = [norm_numeric_lyr]

    for name, input_tensor in input_layer.items():
        if input_tensor.dtype == tf.float32 and len(np.unique(train_in[name])) < max_categorical:
            d_str_selected_uq[name] = np.unique(train_in[name].astype(str)).tolist()
            continue
        if input_tensor.dtype == tf.float32:
            continue
        lookup_layer = tf.keras.layers.StringLookup(vocabulary=np.unique(train_in[name]))
        one_hot_layer = tf.keras.layers.CategoryEncoding(num_tokens=lookup_layer.vocabulary_size())
        lookup_string_lyr = lookup_layer(input_tensor)
        one_hot_string_lyr = one_hot_layer(lookup_string_lyr)
        pre_processed_in_lyr.append(one_hot_string_lyr)
        d_str_selected_uq[name] = np.unique(train_in[name]).tolist()
        # print('np.unique', np.unique(train_in[name]))

    concat_all_lyr = tf.keras.layers.Concatenate()(pre_processed_in_lyr)
    pre_processing_model = tf.keras.Model(input_layer, concat_all_lyr)
    # print('pre_processing_model = ')
    # print(pre_processing_model.summary())

    train_out_l = np.array(train_out)
    # print('train_out_l=', train_out_l)
    target_props = {}
    num_classes = 1
    if out_dtype == tf.float32:
        train_out_ar = np.array(train_out)
        mean_l = np.mean(train_out_ar)
        std_l = np.std(train_out_ar)
        train_out_l = (train_out_ar - mean_l) / std_l
        target_props = {"dtype": str(out_dtype), "mean": mean_l, "std": std_l}

    else:
        lookup_layer = tf.keras.layers.StringLookup()
        lookup_layer.adapt(train_out_l)
        num_classes = lookup_layer.vocabulary_size()
        # print('vocab=', lookup_layer.get_vocabulary())
        # print('class=', num_classes)

        train_out_l = lookup_layer(train_out_l)
        # print('train_out_l=', train_out_l)
        target_props = {"dtype": str(out_dtype), "num_classes": num_classes,
                        "vocab": lookup_layer.get_vocabulary()}

    model = get_computation_model(pre_processing_model, input_layer, learning_rate, num_classes, out_dtype)
    input_train_dist = {name: np.array(value) for name, value in train_in.items()}

    dataset = tf.data.Dataset.from_tensor_slices((input_train_dist, train_out_l))
    # dataset = dataset.shuffle(len(train_out_l), reshuffle_each_iteration=True)
    # train_size = int(0.95 * len(dataset))
    # val_size = len(dataset) - train_size
    # train_dataset = dataset.take(train_size).batch(32)
    # val_dataset = dataset.skip(train_size).take(val_size).batch(32)

    dataset = dataset.shuffle(len(train_out_l))
    val_size = int(0.05 * len(dataset))
    train_dataset = dataset.batch(32)
    val_dataset = dataset.take(val_size).batch(32)

    custom_callback = custom_callbacks(out_dtype)
    # history = model.fit(train_dataset, validation_data=val_dataset, epochs=100)
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=100, callbacks=[custom_callback])
    # draw_history_graph(history, out_dtype)

    # Save the model
    model_name = f'model_{remove_file_ext(os.path.basename(file_path))}.keras'
    # print('d_str_selected_uq=', d_str_selected_uq)
    return model, model_name, target_props, d_str_selected_uq
