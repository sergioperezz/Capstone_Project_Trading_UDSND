
import matplotlib.pyplot as plt
import json
import plotly
import numpy as np
import pandas as pd
import math
from datetime import datetime as dt
from IPython.display import Image, HTML
from flask import Flask
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib

import yfinance as yf
from matplotlib import pyplot
import datetime
from functions import *
from flask_table import Table, Col 
import os
false=False

PLOT_FOLDER = os.path.join('static','plot_photo')

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = PLOT_FOLDER




# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    
    # create visuals
    graphs = []
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 
    print(query)
    start,end,ticker=query.split(" ",2)
    # use model to predict classification for query
    predictions_result, plotName = execute_prediction(start,end,ticker)

    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], plotName+'.jpg')
    # This will render the go.html Please see that file. 
    #return render_template('go.html', query=query,predictions_result=predictions)
    
    return render_template('go.html',query=query,  tables=[predictions_result.to_html(classes='table')], titles=predictions_result.columns.values,user_image=full_filename)
  


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()