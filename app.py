from flask import Flask, request, render_template, jsonify
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
FEATURE_PATH = os.path.join(model_dir, "featureorder.pkl")

# Load model, encoder, scaler, and feature order
try:
    model = joblib.load(MODEL_PATH)
    ohe = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_order = joblib.load(FEATURE_PATH)  # Load saved feature order
    print("✅ Model, encoder, and scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model or transformers: {e}")
    exit(1)
# Define feature categories
categorical_features = ['Venue_State', 'Venue_City']
boolean_features = [
    'Is_Midwest', 'Is_Northeast', 'Is_South', 'Is_West',
    'IsWeekend', 'IsHoliday', 'IsNextDayHoliday', 'Is_outof_CBSA_Area?'
]
numerical_features = [
    'Bin', 'Event_Duration_Hours', 'Total_Tickets_Availability',
    'Average_ticketPrice', 'Max_ticket_price', 'Min_Ticket_Price',
    'Days_Window_Ticket_Sales', 'Total_Number_of_events_shows',
    'Median_Household_Income(2022)', 'Total_Population', 'Male_Population_Total',
    'Total_Male_17_and_Under', 'Total_Male_18_to_29', 'Total_Male_30_to_45',
    'Total_Male_45_to_59', 'Total_Male_60_and_Above', 'Female_Population_Total',
    'Total_Female_17_and_Under', 'Total_Female_18_to_29', 'Total_Female_30_to_45',
    'Total_Female_45_to_59', 'Total_Female_60_and_Above', 'Times_Event_Happened_Here',
    'Number_of_Days_event_hosted'
]

# Initialize Flask app
app = Flask(__name__)

def predict_time_series_for_bins(user_input):
    """Processes user input, applies transformations, ensures correct feature order, and makes predictions."""
    bins = np.arange(0.1, 1.1, 0.1)

    # Convert input dictionary to DataFrame
    new_event_data_template = pd.DataFrame([user_input])

    # Repeat input for each bin
    repeated_data = pd.concat([new_event_data_template] * len(bins), ignore_index=True)
    repeated_data['Bin'] = np.tile(bins, len(new_event_data_template))

    # Ensure all features exist and enforce column order
    for col in feature_order:
        if col not in repeated_data.columns:
            if col in ['Venue_City', 'Venue_State']:
                repeated_data[col] = "Unknown"  # Default category for missing categorical features
            else:
                repeated_data[col] = 0  # Default value for numerical/boolean missing columns

    # Reorder columns to match the training order
    repeated_data = repeated_data[feature_order]

    # Transform categorical features
    new_event_encoded = ohe.transform(repeated_data[['Venue_City', 'Venue_State']])

    # Transform numerical and boolean features
    new_event_scaled = scaler.transform(repeated_data.drop(columns=['Venue_City', 'Venue_State']))

    # Combine transformed features
    X_new = np.hstack([new_event_encoded, new_event_scaled])

    # Predict
    predictions = model.predict(X_new)

    # Create DataFrame for predictions
    results_df = pd.DataFrame({
        'Bin': bins,
        'Predicted_Cumulative_Tickets_Sold': predictions
    })

    return results_df, X_new  # Return both predictions & transformed inputs

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
    transformed_inputs = None  # Initialize transformed inputs
    plot_url = None  # Initialize plot URL

    if request.method == "POST":
        try:
            # Extract user input from form
            user_input = {key: request.form[key] for key in request.form}

            # Convert numerical & boolean inputs properly
            for key in user_input:
                if key in feature_order:  # Ensure only known features are processed
                    if key in ['Venue_City', 'Venue_State']:
                        user_input[key] = str(user_input[key])  # Keep as string
                    elif key in boolean_features:
                        user_input[key] = int(user_input[key])  # Convert booleans (0/1)
                    elif key in numerical_features:
                        user_input[key] = float(user_input[key])  # Convert numbers

            # Predict and get transformed inputs
            results_df, transformed_inputs = predict_time_series_for_bins(user_input)
            plot_url = plot_predictions(results_df)

        except Exception as e:
            return jsonify({"error": f"❌ Error processing request: {e}"}), 400

    return render_template("index.html", plot_url=plot_url, transformed_inputs=transformed_inputs)

if __name__ == "__main__":
    app.run(debug=True)
