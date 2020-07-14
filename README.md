# Capstone Project: Price forecast for Trading 
Repository for the Capstone project in the Data Science Nanodegree of Udacity
## Project motivation

The project shows a process to create an web app where the user can introduce a range dates and some tickers, and the algorithm tries to predict the prices of the Adj close of those tickers in the range.
- 1. Load finance data from yahoo finance.
- 2. Make a Transformation process.
- 3. Clean the data.
- 4. Create a model SVR.
- 5. Evaluate the model with the data.
- 6. Make a prediction on new data.
- 7. Plot the result with Flask.

## Content
```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app
|- functions.py  # file where are all the functions that make the forecast
|- static |-plot_photo # folder to store the plot
-Capture # folder with captures of the app
- README.md
```

## Installation

### Dependencies
- Python 3.5+
- Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
- Finance libraries: yfinance
- date Libraries: datime
- Web App and Data Visualization: Flask, Plotly, matplotlib

### User installation
Clone the repository:
```
https://github.com/sergioperezz/Disaster-response-Pipeline-UND.git
```
## Running the tests 

```
1. Run the following command in the app's directory to run your web app.
    `python run.py`

2. Go to http://localhost:3001/
```

## Authors

[Sergio Pérez](https://github.com/sergioperezz)

## License and Acknowledgements

Project developed under MIT licensing.
[developers.arcgis.com](https://developers.arcgis.com/python)
[stackoverflow](https://es.stackoverflow.com/)
[sarahleejane](sarahleejane.github.io)
## Capture

![alt text](https://github.com/sergioperezz/Capstone_Project_Trading_UDSND/blob/master/Capture/Overview.PNG)
![alt text](https://github.com/sergioperezz/Capstone_Project_Trading_UDSND/blob/master/Capture/Result1.PNG)
![alt text](https://github.com/sergioperezz/Capstone_Project_Trading_UDSND/blob/master/Capture/plot.PNG)
