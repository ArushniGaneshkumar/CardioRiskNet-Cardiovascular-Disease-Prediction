import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input
from tensorflow.keras.optimizers import Adam

def build_model(input_shape):
    """
    Builds and returns the CardioRiskNet model.

    Parameters:
        input_shape (int): Number of features (input dimension)

    Returns:
        model (tf.keras.Model): Compiled LSTM-based model
    """
    model = Sequential([
        Input(shape=(1, input_shape)),  # Shape: (batch_size, time_steps, features)
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='linear')  # Output is a continuous risk score
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model
