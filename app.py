from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Ensure the saved_models directory exists
model_dir = "saved_model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Define paths for saved models
MODEL_PATH = os.path.join(model_dir, "best_model1.pkl")
ENCODER_PATH = os.path.join(model_dir, "encoder.pkl")
SCALER_PATH = os.path.join(model_dir, "scaler.pkl")

# Load model, encoder, and scaler
try:
    model = joblib.load(MODEL_PATH)
    ohe = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model, encoder, and scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model or transformers: {e}")
    exit(1)

# Define features
categorical_features = ['Venue_State', 'Venue_City', 'Year']
boolean_features = [
    'Is_Midwest', 'Is_Northeast', 'Is_South', 'Is_West',
    'IsWeekend', 'IsHoliday', 'IsNextDayHoliday', 'Is_outof_CBSA_Area?'
]
numerical_features = [
    'Bin', 'Event_Duration_Hours', 'Total_Tickets_Availability',
    'Average_ticketPrice', 'Max_ticket_price', 'Min_Ticket_Price',
    'Days_Window_Ticket_Sales', 'Total_Number_of_events_shows',
    'Median_Household_Income(2022)', 'Total_Population', 'Male_Population_Total',
    'Female_Population_Total', 'Times_Event_Happened_Here',
    'Number_of_Days_event_hosted'
]

# Initialize Flask app
app = Flask(__name__)

def predict_time_series_for_bins(user_input):
    """Processes user input, scales & encodes it, and makes predictions."""
    bins = np.arange(0.1, 1.1, 0.1)

    # Convert input dictionary to DataFrame
    new_event_data_template = pd.DataFrame([user_input])

    # Repeat input for each bin
    repeated_data = pd.concat([new_event_data_template] * len(bins), ignore_index=True)
    repeated_data['Bin'] = np.tile(bins, len(new_event_data_template))

    # Ensure all columns are present
    for col in numerical_features + boolean_features:
        if col not in repeated_data.columns:
            repeated_data[col] = 0  # Default value for missing numerical/boolean

    for col in categorical_features:
        if col not in repeated_data.columns:
            repeated_data[col] = "Unknown"  # Default category

    # Transform input
    new_event_encoded = ohe.transform(repeated_data[categorical_features])
    new_event_scaled = scaler.transform(repeated_data[numerical_features + boolean_features])
    X_new = np.hstack([new_event_encoded, new_event_scaled])

    # Predict
    predictions = model.predict(X_new)

    # Create DataFrame with results
    results_df = pd.DataFrame({
        'Bin': bins,
        'Predicted_Cumulative_Tickets_Sold': predictions
    })

    return results_df

def plot_predictions(results_df):
    """Generates a plot and returns the image as a base64 string."""
    plt.figure(figsize=(10, 5))
    plt.plot(results_df['Bin'], results_df['Predicted_Cumulative_Tickets_Sold'], marker='o', color='b', label="Predicted Sales")
    plt.xlabel("Time Bin")
    plt.ylabel("Cumulative Tickets Sold")
    plt.title("Predicted Ticket Sales Over Time")
    plt.legend()
    plt.grid(True)

    # Save plot as an image
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return img_base64

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Extract user input
        user_input = {
            'Venue_City': request.form['Venue_City'],
            'Venue_State': request.form['Venue_State'],
            'Year': int(request.form['Year']),
            'Is_Midwest': int(request.form.get('Is_Midwest', 0)),
            'Is_Northeast': int(request.form.get('Is_Northeast', 0)),
            'Is_South': int(request.form.get('Is_South', 0)),
            'Is_West': int(request.form.get('Is_West', 0)),
            'IsWeekend': int(request.form.get('IsWeekend', 0)),
            'IsHoliday': int(request.form.get('IsHoliday', 0)),
            'IsNextDayHoliday': int(request.form.get('IsNextDayHoliday', 0)),
            'Is_outof_CBSA_Area?': int(request.form.get('Is_outof_CBSA_Area?', 0)),
            'Event_Duration_Hours': float(request.form['Event_Duration_Hours']),
            'Total_Tickets_Availability': int(request.form['Total_Tickets_Availability']),
            'Average_ticketPrice': float(request.form['Average_ticketPrice']),
            'Max_ticket_price': float(request.form['Max_ticket_price']),
            'Min_Ticket_Price': float(request.form['Min_Ticket_Price']),
            'Days_Window_Ticket_Sales': int(request.form['Days_Window_Ticket_Sales']),
            'Total_Number_of_events_shows': int(request.form['Total_Number_of_events_shows']),
            'Median_Household_Income(2022)': float(request.form['Median_Household_Income(2022)']),
            'Total_Population': int(request.form['Total_Population']),
            'Male_Population_Total': int(request.form['Male_Population_Total']),
            'Female_Population_Total': int(request.form['Female_Population_Total']),
            'Times_Event_Happened_Here': int(request.form['Times_Event_Happened_Here']),
            'Number_of_Days_event_hosted': int(request.form['Number_of_Days_event_hosted'])
        }

        # Predict
        results_df = predict_time_series_for_bins(user_input)
        plot_url = plot_predictions(results_df)

        return render_template("index.html", plot_url=plot_url)

    return render_template("index.html", plot_url=None)

if __name__ == "__main__":
    app.run(debug=True)
