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
import json
import urllib
import base64

model = tf.keras.models.load_model('sentiment_analysis_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def analyze_text(text):
    text_sequence = tokenizer.texts_to_sequences([text])
    text_sequence = pad_sequences(text_sequence, maxlen=100)
    predicted_rating = model.predict(text_sequence)[0]

    sentiment = 'Negative' if np.argmax(predicted_rating) == 0 else 'Positive'
    return sentiment

def decode_request(event, data):
    try:
        # Decode from Base64
        decoded_data = base64.b64decode(data).decode()
        data = json.loads(decoded_data)
    except json.JSONDecodeError as json_err:
        return {
            f'error with that request {json_err} data: {data}'
        }
    except base64.binascii.Error as b64_err:
        return {
            f'error with that request {b64_err} data: {data}'
        }
    except Exception as e:
        return {
            f'error with that request {e} data: {data}'
        }
    return data
    
def lambda_handler(event, context):
    
    # Parse the input data
    # Attempt to parse the query_data as JSON
    queryType = event.get('queryType', '')
    query = event.get('query', '')

    if queryType == 'single':
        sentiment = analyze_text(query)
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'text'},
            'body': sentiment
        }
    elif queryType == 'multiple':
        decoded_query = decode_request(event, query)
        results = []
        for text in decoded_query:
            sentiment = analyze_text(text)
            results.append({'text': text, 'sentiment': sentiment})
        # Convert results to a CSV format
        output = io.StringIO()
        pd.DataFrame(results).to_csv(output, index=False)
        csv_output = output.getvalue()
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'text/csv'},
            'body': csv_output
        }