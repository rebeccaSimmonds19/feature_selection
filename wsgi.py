from sklearn.model_selection import train_test_split
from catboost import Pool, CatBoostRegressor, cv
import pandas as pd
import plotly.offline as offline
import plotly.graph_objs as go
import seaborn as sns
import os
from os import environ
import shutil
import argparse
from flask import Flask, render_template
import json

application = Flask(__name__)

@application.route('/')
def index():

    data = pd.read_csv("wine-reviews/winemag-data_first150k.csv")
    X = data.drop(columns=['points'])
    X=X.fillna(-1)
    categorical_features_indices =[0,1,2,3,4,5,6,7,8,9]
    y=data['points']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=52)
    model=perform_model(X_train, y_train,X_valid, y_valid,X_test, y_test, categorical_features_indices =[0,1,2,3,4,5,6,7,8,9])
    feature_score = pd.DataFrame(list(zip(X.dtypes.index, model.get_feature_importance(Pool(X, label=y, cat_features=categorical_features_indices)))),
                    columns=['Feature','Score'])

    feature_score = feature_score.sort_values(by='Score', ascending=False, inplace=False, kind='quicksort', na_position='last')

    data = [go.Bar(
            x=feature_score.Feature,
            y=feature_score.Score,
            marker=dict(
            color='purple'
    ) )]
    layout = go.Layout(
        autosize=True,
        title="feature selection"
        )

    fig = go.Figure(data=data, layout=layout)
    template = offline.plot(fig)
    print(template)
    return make_template()

def perform_model(X_train, y_train,X_valid, y_valid,X_test, y_test, categorical_features_indices):
    model = CatBoostRegressor(
        random_seed = 400,
        loss_function = 'RMSE',
        iterations=400,
    )

    trained_model = model.fit(
        X_train, y_train,
        cat_features = categorical_features_indices,
        eval_set=(X_valid, y_valid),
        verbose=False
    )
    print("RMSE on training data: "+ model.score(X_train, y_train).astype(str))
    print("RMSE on test data: "+ model.score(X_test, y_test).astype(str))
    return model


def make_template():
    # make the templates dir
    new_path = '/opt/app-root/src/templates'
    if not os.path.exists(new_path):
        os.makedirs(new_path)
        # move the file to the templates dir
        shutil.move('/opt/app-root/src/temp-plot.html', new_path)
    return render_template("temp-plot.html", title='Catboost Feature Importance Ranking')

if __name__ == '__main__':
    application.run()
