import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

model = tf.keras.models.load_model('sentiment_analysis_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def lambda_handler(event, context):
    # Tokenize and pad the input text
    print(event["query"])
    text = event["query"]
    text_sequence = tokenizer.texts_to_sequences([text])
    text_sequence = pad_sequences(text_sequence, maxlen=100)

    # Make a prediction using the trained model
    predicted_rating = model.predict(text_sequence)[0]

    if np.argmax(predicted_rating) == 0:
        return f'{event}: Negative'
    else:
        return f'{event}: positive'