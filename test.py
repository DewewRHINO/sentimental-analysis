import requests
import json
import base64

def send_single_query(endpoint, text):
    payload = {
        'queryType': 'single',
        'query': text
    }
    response = requests.get(endpoint, params=payload)
    return response.json()

def send_multiple_query(endpoint, texts):
    json_data = json.dumps(texts)
    base64_encoded_data = base64.b64encode(json_data.encode()).decode()

    payload = {
        'queryType': 'multiple',
        'query': base64_encoded_data
    }
    response = requests.get(endpoint, params=payload)
    return response.json()

# Endpoint URL
api_endpoint = "https://bbtflv6yqf.execute-api.us-east-1.amazonaws.com/Initial/sentimental-analysis"

# Test single query
single_query_response = send_single_query(api_endpoint, "This is a great movie")
print("Single Query Response:", single_query_response)

# Test multiple query
multiple_query_response = send_multiple_query(api_endpoint, ["This is a great movie", "This is a terrible movie", "I am not sure about this"])
print("Multiple Query Response:", multiple_query_response)
