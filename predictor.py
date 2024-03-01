import pandas as pd
from pycaret.classification import predict_model, load_model

def load_data(filepath):

    df = pd.read_csv(filepath, index_col='customerID')
    return df


def make_predictions(df):
    model = load_model('churn_model')
    predictions = predict_model(model, data=df)
    return predictions


if __name__ == "__main__":
    df = load_data('new_churn_data.csv')
    predictions = make_predictions(df)
    print('predictions:')
    print(predictions)