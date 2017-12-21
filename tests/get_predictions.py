"""This script should be run on your local machine."""

import requests
import pickle
import numpy as np

API_HOST = 'http://18.217.196.13:5000'

PREDICT_API = '/predict'

def load_data():

    with open("test_data.pkl",'rb') as f:
        test_data = pickle.load(f)
    return np.array(test_data).tolist()

def predict(test_data_file):
    print("Trying predict endpoint...")
    # Note that this is a POST request as we need to send the
    # passenger data to the server.
    # The requests library converts the passenger data into
    # JSON before sending it over. This is because the server
    # expects to receive the passenger data in the form of a JSON.
    r = requests.post(API_HOST + PREDICT_API,
                      json=test_data_file)

    # Also note that we're now using r.json(), not r.text.
    # This is because the server sends its response back as a
    # JSON object, which needs to be decoded by the requests
    # library.
    if r.status_code == 200:
        print("Success!")
        print(r.json())
    else:
        print("Status code indicates a problem:", r.status_code)

def main():
    data = load_data()
    predict(data)

# Entry point for application (i.e. program starts here)
if __name__ == '__main__':
    main()