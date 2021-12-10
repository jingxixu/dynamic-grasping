import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl


def create_model(input_shape, output_shape, sequence_len=None, stateful=False, batch_size=None):
    simple_lstm_model = tfk.models.Sequential([
        tfkl.Input(batch_shape=(batch_size, sequence_len,) + input_shape),
        tfkl.LSTM(100, return_sequences=True, stateful=stateful),
        tfkl.Dense(100),
        tfkl.Dropout(rate=0.1),
        tfkl.Dense(output_shape),
    ])
    return simple_lstm_model