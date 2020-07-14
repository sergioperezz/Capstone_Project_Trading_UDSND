
import matplotlib.pyplot as plt

import numpy as np
from numpy import array
import pandas as pd
import math
from datetime import datetime as dt
from IPython.display import Image, HTML

from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

import yfinance as yf
from matplotlib import pyplot
import datetime
import os

def get_prices(comp):
    """
    Function that gets the close, Open, high, low, Adj close and volume of the tickers. It Returns a dataframe with the Adj Close of the tickers
    Args: Company tickers to predict in format "AMZN AAPL"
    return: Dataframe with Adj Close of past 10 years of the companies

    """
    
    try: 
        fi_data = yf.download(comp,period="10y")

        if " " in comp:
            return fi_data['Adj Close']

        else:
            return pd.DataFrame({'Date':fi_data['Adj Close'].index, comp :fi_data['Adj Close'].values}).set_index("Date")
    except:
        print("Incorrect input ticker.Please, introduce the company tickers between spaces: 'AMZN GOOGL' ")
 

def create_dates(start,days):
    """
    create future forecast dates
    Args: start date,  number of days to forecast 
    return: DataFrame with number of days and dates
    """ 
    v = pd.date_range(start=start, periods=days+1, freq='D', closed='right')
    day_to_forecast = pd.DataFrame(index=v) 
    return day_to_forecast


def get_value_name(tickers,i):
    """
    get values, tickers name and drop null values
    Args: tickers, index
    return: ticker value, ticker name
    """ 
    ticker_value = tickers[[tickers.columns[i]]].dropna()
    ticker_name = tickers.columns[i]
    return ticker_value, ticker_name 


def train_test_split(value, name, ratio):
    """
    train-test split for a user input ratio
    Args: value, ticker name, ratio
    return: train block, test block, split row
    """ 
    nrow = len(value)
    print(name+' total samples: ',nrow)
    split_row = int((nrow)*ratio)
    print('Training samples: ',split_row)
    print('Testing samples: ',nrow-split_row)
    train = value.iloc[:split_row]
    test = value.iloc[split_row:]
    return train, test, split_row 


def data_transformation(train_tract1,test_tract1):
    """
    data transformation. Get the train and test parts and transform them with the minMaxScaler to normalize them.
    Args: train and test
    Return: tran and test section transformed
    """
    
    scaler = MinMaxScaler()
    train_tract1_scaled = scaler.fit_transform(train_tract1)
    test_tract1_scaled = scaler.fit_transform(test_tract1)          
    train_tract1_scaled_df = pd.DataFrame(train_tract1_scaled, index = train_tract1.index, columns=[train_tract1.columns[0]])
    test_tract1_scaled_df = pd.DataFrame(test_tract1_scaled,
                                         index = test_tract1.index, columns=[test_tract1.columns[0]])
    return train_tract1_scaled_df, test_tract1_scaled_df, scaler 


def timeseries_feature_builder(df, lag):
    """
    feature builder - This section creates feature set with lag number of predictors--Creating features using lagged data
    Args: dataframe, and lag days
    return: the dataframe with the new data
    """
    df_copy = df.copy()
    for i in range(1,lag):
        df_copy['lag'+str(i)] = df.shift(i) 
    return df_copy
    df_copy = df.copy()
    
    
def make_arrays(train_tract1,test_tract1):
    """
    preprocessing -- drop null values and make arrays 
    Args: train and test blocks
    return: matrix of train and test
    """
    X_train_tract1_array = train_tract1.dropna().drop(train_tract1.columns[0], axis=1).values
    y_train_tract1_array = train_tract1.dropna()[train_tract1.columns[0]].values
    X_test_tract1_array = test_tract1.dropna().drop(test_tract1.columns[0], axis=1).values
    y_test_tract1_array = test_tract1.dropna()[test_tract1.columns[0]].values    
    return X_train_tract1_array, y_train_tract1_array, X_test_tract1_array, y_test_tract1_array    


# 
def fit_svr(X_train_tract1_array, y_train_tract1_array, X_test_tract1_array, y_test_tract1_array):
    """
    fitting & Validating using SVR
    Args: matrix: X_train, Y_train, X_test, Y_test
    return: model, y pred_test
    """
   
    model_svr = SVR(kernel='rbf', gamma='auto', tol=0.001, C=10.0, epsilon=0.001)
    model_svr.fit(X_train_tract1_array,y_train_tract1_array)
    y_pred_train_tract1 = model_svr.predict(X_train_tract1_array)
    y_pred_test_tract1 = model_svr.predict(X_test_tract1_array)        
    print('r-square_SVR_Test: ', round(model_svr.score(X_test_tract1_array,y_test_tract1_array),2))
    return model_svr, y_pred_test_tract1 


def valid_result_svr(scaler, y_pred_test_tract1, ticker_value, split_row, lag):
    """
    validation result
    Args: Scaler(to make inverse transform), Y_test, ticker, splot row, lag
    return test_tract1_pred( Adj close prediction)
    """
    new_test_tract1 = ticker_value.iloc[split_row:]
    test_tract1_pred = new_test_tract1.iloc[lag:].copy()
    y_pred_test_tract1_transformed = scaler.inverse_transform([y_pred_test_tract1])
    y_pred_test_tract1_transformed_reshaped = np.reshape(y_pred_test_tract1_transformed,(y_pred_test_tract1_transformed.shape[1],-1))
    test_tract1_pred['Forecast'] = np.array(y_pred_test_tract1_transformed_reshaped)
    return test_tract1_pred


def forecast_svr(X_test_tract1_array, days ,model_svr, lag, scaler):
    """
    multi-step future forecast. for each ticker, forescast
    Args: X_test_tract1_array, days ,model_svr, lag, scaler
    return 
    """
    last_test_sample = X_test_tract1_array[-1]        
    X_last_test_sample = np.reshape(last_test_sample,(-1,X_test_tract1_array.shape[1]))        
    y_pred_last_sample = model_svr.predict(X_last_test_sample)                
    new_array = X_last_test_sample
    new_predict = y_pred_last_sample
    new_array = X_last_test_sample
    new_predict = y_pred_last_sample

    days_svr=[]
    for i in range(0,days):               
            new_array = np.insert(new_array, 0, new_predict)                
            new_array = np.delete(new_array, -1)
            new_array_reshape = np.reshape(new_array, (-1,lag))                
            new_predict = model_svr.predict(new_array_reshape)
            temp_predict = scaler.inverse_transform([new_predict])
            days_svr.append(temp_predict[0][0].round(2))
            
    return days_svr 


def company_close_svr(tickers, lag,start_day, days):     
    """
    function to execute the full app. iterate over the tickers, makes the preprocessing, fitting, predictiong and evaluation
    Args: tickers, lag(numbers of previous days), start date, and days to forecast
    return: prediction, plot name to put in the web app.
    
    """
    
    day_to_forecast_svr = create_dates(start_day,days)
    
    dir_name = "static/plot_photo/"
    test = os.listdir(dir_name)

    for item in test:
        if item.endswith(".jpg"):
            os.remove(os.path.join(dir_name, item))
        
        
    if len(tickers.columns)>1:
        fig, ax = plt.subplots(len(tickers.columns),1,figsize=(15,10),sharex=True)
    else:
        fig, axi = plt.subplots(len(tickers.columns),1,figsize=(15,10),sharex=True)
        ax=array([axi])
        
    for i in range(len(tickers.columns)):
        
        # preprocessing
        ticker_value, ticker_name = get_value_name(tickers,i)       
        train_tract1, test_tract1, split_row = train_test_split(ticker_value, ticker_name, 0.80)              
        train_tract1_scaled_df, test_tract1_scaled_df, scaler = data_transformation(train_tract1,test_tract1)        
        train_tract1 = timeseries_feature_builder(train_tract1_scaled_df,lag+1)
        test_tract1 = timeseries_feature_builder(test_tract1_scaled_df, lag+1)        
        X_train_tract1_array, y_train_tract1_array, X_test_tract1_array, y_test_tract1_array = make_arrays(train_tract1,
                                                                                                           test_tract1)

        # SVR modeling
        model_svr, y_pred_test_tract1 = fit_svr(X_train_tract1_array, y_train_tract1_array,
                                                X_test_tract1_array, y_test_tract1_array)                       
        test_tract1_pred = valid_result_svr(scaler, y_pred_test_tract1, ticker_value, split_row, lag)        
        days_svr = forecast_svr(X_test_tract1_array, days, model_svr, lag, scaler)            
        day_to_forecast_svr[ticker_name] = np.array(days_svr)        
        
        # plot result
        

        #plt.figure(figsize=(20,5))
        ax[i].plot(test_tract1_pred)
        ax[i].plot(day_to_forecast_svr[ticker_name], color='red', label='forecast') 
        ax[i].set_ylabel('Value ($)')
        ax[i].legend(loc='upper left')
        ax[i].set_title(ticker_name + ' Forecast')
  
    
    plotName=str(datetime.datetime.now().strftime("%d%m%Y%H%M%S"))
    fig.savefig("static/plot_photo/"+plotName+".jpg")
    
        
    return(day_to_forecast_svr),plotName



def execute_prediction(start_date, end_date,tickers):
    """
    query constructor. Here the data will be transformed to be in the correct format 
    Args: start date, end date, tickers (info that the user will provide)
    return: the same of company_close_svr
    """
    
    today=datetime.date.today()
    day=datetime.timedelta(days=1)
    start_date_dt=datetime.datetime.strptime(start_date, '%Y-%m-%d').date()  
    end_date_dt=datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
    
    if start_date_dt<today:
        return "First day must be (at least) today. Please, set other first day"

    if abs((start_date_dt-end_date_dt).days)>31:
        return "Cannot forecast more than 31 days. Please, reduce the number of days to forecast"
   
    if end_date_dt< start_date_dt:
        return "End day connot be before start day"  
    
    data=get_prices(tickers)
    
    start_date_dt=datetime.datetime.strptime(start_date, '%Y-%m-%d').date()-day   
    return company_close_svr(data, 120, start_date_dt,abs((start_date_dt-end_date_dt).days))
    




