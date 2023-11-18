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
import io
import pandas as pd

model = tf.keras.models.load_model('sentiment_analysis_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def lambda_handler(event, context):
    # Check if the input is a single string or multiple items
    if isinstance(event['query'], str):
        texts = [event['query']]
    elif isinstance(event['query'], list):
        texts = event['query']
    else:
        return "Invalid input format"

    # Process each text
    results = []
    for text in texts:
        text_sequence = tokenizer.texts_to_sequences([text])
        text_sequence = pad_sequences(text_sequence, maxlen=100)
        predicted_rating = model.predict(text_sequence)[0]

        sentiment = 'Negative' if np.argmax(predicted_rating) == 0 else 'Positive'
        results.append({'text': text, 'sentiment': sentiment})

    # Convert results to a CSV format
    if len(results) == 1:
        return f"{results[0]['text']}, {results[0]['sentiment']}"
    else:
        output = io.StringIO()
        pd.DataFrame(results).to_csv(output, index=False)
        return output.getvalue()

# Example usage
event_single = {'query': 'Good idea'}
event_multiple = {'query': ['Good idea', 'This is a terrible movie', 'I am not sure about this']}

# For a single input
print(lambda_handler(event_single, None))

# For multiple inputs
print(lambda_handler(event_multiple, None))
