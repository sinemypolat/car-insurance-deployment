import pandas as pd
import numpy as np

MODEL_DIRECTORY = 'model'
MODEL_FILE_NAME = '%s/model_file.pkl' % MODEL_DIRECTORY


def predict(input_df, model):
    print("Input data frame is...\n")
    print("-----------")
    print(input_df.head())
    print("-----------")

    input_array = np.array(input_df)

    predictions = model.predict(input_array).tolist()
    predictions = [int(prediction) for prediction in predictions]

    return {'predictions': predictions}