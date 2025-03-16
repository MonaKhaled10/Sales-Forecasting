from flask import Flask, request, render_template, jsonify
import pandas as pd
from prophet import Prophet
import os

app = Flask(__name__)

# Define Egyptian Holidays Once
egypt_holidays = pd.DataFrame({
    'holiday': 'egypt_holidays',
    'ds': pd.to_datetime([
        "2023-01-07", "2023-04-21", "2023-04-22", "2023-04-23",
        "2023-06-27", "2023-06-28", "2023-06-29", "2023-06-30",
        "2023-07-23", "2023-09-27",
        "2024-01-07", "2024-04-10", "2024-04-11", "2024-04-12",
        "2024-06-16", "2024-06-17", "2024-06-18", "2024-06-19",
        "2024-07-23", "2024-09-15"
    ]),
    'lower_window': 0,
    'upper_window': 1
})

def preprocess_data(file):
    """Preprocess CSV File for Prophet Model"""
    try:
        # Ensure it's a CSV file
        if not file.filename.endswith('.csv'):
            return None, "Invalid file format. Please upload a CSV file."

        df = pd.read_csv(file.stream)

        # Ensure required columns exist
        if not {'Date', 'SalesValue'}.issubset(df.columns):
            return None, "Missing required columns: 'Date' and 'SalesValue'."

        # Convert to datetime and sort by date
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date', 'SalesValue'])  # Remove rows with NaN values
        df = df.sort_values(by='Date')

        # Aggregate sales per week
        df_weekly = df.groupby(pd.Grouper(key='Date', freq='W')).agg({'SalesValue': 'sum'}).reset_index()
        df_weekly.rename(columns={'Date': 'ds', 'SalesValue': 'y'}, inplace=True)

        return df_weekly, None
    except Exception as e:
        return None, str(e)


@app.route('/', methods=['GET', 'POST'])
def home():
    predictions = None
    error = None

    if request.method == 'POST':
        file = request.files.get('file')
        
        if not file or file.filename == '':
            error = "No file uploaded or selected."
        else:
            df_weekly, error = preprocess_data(file)

            if df_weekly is not None:
                try:
                    # Define Prophet model
                    model = Prophet(
                        yearly_seasonality=True,
                        weekly_seasonality=True,
                        daily_seasonality=False,
                        seasonality_mode='multiplicative',
                        changepoint_prior_scale=0.5,
                        seasonality_prior_scale=0.1,
                        holidays=egypt_holidays
                    )
                    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
                    model.fit(df_weekly)

                    # Make predictions for the next 4 weeks
                    future = model.make_future_dataframe(periods=4, freq='W')
                    forecast = model.predict(future)

                    # Extract last 4 weeks of predictions
                    predictions = forecast[['ds', 'yhat']].tail(4).to_dict(orient='records')

                except Exception as e:
                    error = str(e)

    return render_template('index.html', predictions=predictions, error=error)


@app.route('/predict', methods=['POST'])
def predict():
    """API Endpoint to return predictions as JSON"""
    file = request.files.get('file')

    if not file or file.filename == '':
        return jsonify({"error": "No file uploaded"}), 400

    df_weekly, error = preprocess_data(file)

    if df_weekly is None:
        return jsonify({"error": error}), 400

    try:
        # Define Prophet model
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, holidays=egypt_holidays)
        model.fit(df_weekly)

        # Make predictions for the next 4 weeks
        future = model.make_future_dataframe(periods=4, freq='W')
        forecast = model.predict(future)

        # Return JSON response
        predictions = forecast[['ds', 'yhat']].tail(4).to_dict(orient='records')
        return jsonify(predictions)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
