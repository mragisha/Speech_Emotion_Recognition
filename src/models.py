"""Model factory: LSTM, BiLSTM, GRU, BiGRU."""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, GRU, Bidirectional, Dense, Dropout
)
import config


def _base_head(model: Sequential, n_classes: int) -> Sequential:
    model.add(Dropout(config.DROPOUT_RATE))
    for units in config.DENSE_UNITS:
        model.add(Dense(units, activation="relu"))
        model.add(Dropout(config.DROPOUT_RATE))
    model.add(Dense(n_classes, activation="softmax"))
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    return model


def build_lstm(n_classes: int, input_shape=(config.N_MFCC, 1)) -> Sequential:
    model = Sequential(name="LSTM")
    model.add(LSTM(config.LSTM_UNITS, return_sequences=False, input_shape=input_shape))
    return _base_head(model, n_classes)


def build_bilstm(n_classes: int, input_shape=(config.N_MFCC, 1)) -> Sequential:
    model = Sequential(name="BiLSTM")
    model.add(Bidirectional(LSTM(config.LSTM_UNITS, return_sequences=False),
                             input_shape=input_shape))
    return _base_head(model, n_classes)


def build_gru(n_classes: int, input_shape=(config.N_MFCC, 1)) -> Sequential:
    model = Sequential(name="GRU")
    model.add(GRU(config.LSTM_UNITS, return_sequences=False, input_shape=input_shape))
    return _base_head(model, n_classes)


def build_bigru(n_classes: int, input_shape=(config.N_MFCC, 1)) -> Sequential:
    model = Sequential(name="BiGRU")
    model.add(Bidirectional(GRU(config.LSTM_UNITS, return_sequences=False),
                             input_shape=input_shape))
    return _base_head(model, n_classes)


MODEL_REGISTRY = {
    "lstm":   build_lstm,
    "bilstm": build_bilstm,
    "gru":    build_gru,
    "bigru":  build_bigru,
}
