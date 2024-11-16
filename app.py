import gradio as gr
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Load the saved model and preprocessors
model = load_model('lstm_model.h5')
cases_scaler = joblib.load('scaler_encoder.save')
label_encoder = joblib.load('location_encoder.save')

# Get the input shape from the model directly
seq_length = model.get_layer(index=0).input_shape[0][1]  # This will get the actual sequence length from the model

def create_sequences_with_location(cases_data, location_data, seq_length):
    """Convert input data into sequences with location information"""
    X_cases = []
    X_location = []
    
    for i in range(len(cases_data) - seq_length):
        X_cases.append(cases_data[i:i + seq_length])
        X_location.append(location_data[i + seq_length - 1])
    
    return np.array(X_cases), np.array(X_location)

def process_input_data(input_data, location):
    """Process input data string and location"""
    # Input validation
    if not input_data or not location:
        return None, None, "Error: Please provide both input data and location"
    
    try:
        # Convert string input to list of numbers
        data = [float(x.strip()) for x in input_data.split(',')]
        
        # Validate location
        if location not in label_encoder.classes_:
            return None, None, f"Error: Location '{location}' not found in training data. Available locations: {', '.join(label_encoder.classes_)}"
        
        location_encoded = label_encoder.transform([location])[0]
        return data, location_encoded, None
        
    except ValueError as e:
        return None, None, "Error: Please ensure input is comma-separated numbers"

def format_date(base_date, steps):
    """Format date string for x-axis labels"""
    try:
        return (datetime.strptime(base_date, '%Y-%m-%d') + timedelta(weeks=steps)).strftime('%Y-%m-%d')
    except ValueError:
        return f"Week {steps + 1}"  # Fallback if date is invalid

def generate_prediction(input_data, location, base_date, num_weeks):
    """Generate weekly predictions using the loaded model"""
    # Input validation
    if not location:
        return None, None, "Error: Please select a location"
    
    # Process input data
    data, location_encoded, error = process_input_data(input_data, location)
    if error:
        return None, None, error
    
    if len(data) < seq_length:
        return None, None, f"Error: Need at least {seq_length} data points (weeks) for prediction. You provided {len(data)}. Please provide {seq_length} weeks of historical data."
    
    try:
        # Scale the cases data
        scaled_data = cases_scaler.transform(np.array(data).reshape(-1, 1))
        
        # Generate predictions
        predictions = []
        current_sequence = scaled_data[-seq_length:]  # Make sure we use the correct sequence length
        current_location = location_encoded
        
        # Predict for specified number of weeks
        for _ in range(num_weeks):
            X_cases = current_sequence.reshape(1, seq_length, 1)  # Reshape with correct sequence length
            X_location = np.array([current_location]).reshape(1, 1)
            
            pred = model.predict([X_cases, X_location], verbose=0)
            predictions.append(pred[0, 0])
            
            current_sequence = np.vstack((current_sequence[1:], pred))
        
        # Inverse transform predictions
        predictions = cases_scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        # Create date labels
        dates = [format_date(base_date, i) for i in range(num_weeks)]
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(data)), data, label='Historical Data', color='blue')
        plt.plot(range(len(data)-1, len(data) + len(predictions)-1), 
                 predictions, label='Predictions', color='red', linestyle='--')
        
        plt.axvline(x=len(data)-1, color='gray', linestyle='--', alpha=0.5)
        
        plt.title(f'Dengue Cases Prediction for {location}')
        plt.xlabel('Time')
        plt.ylabel('Weekly Cases')
        plt.legend()
        plt.grid(True)
        
        plt.xticks(rotation=45)
        
        pred_start = len(data) - 1
        for i, date in enumerate(dates):
            plt.annotate(date, (pred_start + i, predictions[i][0]),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.tight_layout()
        plt.savefig('temp_plot.png')
        plt.close()
        
        prediction_text = "\n".join([f"Week {i+1} ({dates[i]}): {pred[0]:.0f} cases" 
                                    for i, pred in enumerate(predictions)])
        
        return 'temp_plot.png', prediction_text, None
        
    except Exception as e:
        return None, None, f"Error during prediction: {str(e)}"

def create_gradio_interface():
    """Create and launch the Gradio interface"""
    # Get list of available locations
    available_locations = sorted(list(label_encoder.classes_))
    
    # Set default location to first available location
    default_location = available_locations[0] if available_locations else None
    
    # Create example input with correct number of data points
    example_data = ", ".join([str(100 + i * 10) for i in range(seq_length)])
    
    input_description = f"""
    Enter exactly {seq_length} historical weekly case counts as comma-separated numbers.
    Example: {example_data}
    """
    
    interface = gr.Interface(
        fn=generate_prediction,
        inputs=[
            gr.Textbox(
                label="Historical Weekly Cases",
                placeholder=f"Enter {seq_length} comma-separated numbers",
                info=input_description
            ),
            gr.Dropdown(
                choices=available_locations,
                label="Location",
                value=default_location,
                info="Select the location for prediction"
            ),
            gr.Textbox(
                label="Base Date (YYYY-MM-DD)",
                placeholder="2024-01-01",
                value="2024-01-01",
                info="Enter the date of the last historical data point"
            ),
            gr.Slider(
                minimum=1,
                maximum=4,
                step=1,
                label="Number of weeks to predict",
                value=2
            )
        ],
        outputs=[
            gr.Image(label="Prediction Plot"),
            gr.Textbox(label="Weekly Predictions"),
            gr.Textbox(label="Error Messages")
        ],
        title="Dengue Cases Prediction by Location",
        description=f"""
        This application predicts weekly dengue cases for specific locations using a trained LSTM model.
        Please provide exactly {seq_length} weeks of historical case counts for accurate predictions.
        """,
        examples=[
            [example_data, available_locations[0], "2024-01-01", 2],
            [", ".join([str(80 + i * 2) for i in range(seq_length)]), available_locations[-1], "2024-01-01", 4]
        ],
        theme=gr.themes.Soft()
    )
    
    return interface

# Launch the application
if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(share=True)