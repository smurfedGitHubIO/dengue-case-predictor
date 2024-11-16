# dengue-case-predictor

A Gradio application that predicts dengue cases for the next 1-4 weeks based on historical data. This application allows users to input relevant factors, and it will predict the number of dengue cases expected in the upcoming weeks.

## Prerequisites

Before running the application, ensure you have the following dependencies installed:

- Python 3.x
- Gradio
- Any other libraries specified in `requirements.txt`

You can install all dependencies by running:

```bash
pip install -r requirements.txt
```

## Running the application

To run the application, follow these steps:

1. Clone this repository or download the source code.
2. Navigate to the directory containing app.py.
3. Run the application using the following command:

```bash
python app.py
```

Once the app is running, it will provide a web interface where you can input data and receive predictions for dengue cases in the next 1-4 weeks.

## Features
- **Predict dengue cases**: Provides predictions based on historical data for 1-4 weeks.
- **User input**: Users can input different factors related to dengue cases, and the model will provide predictions.
- **Gradio interface**: Interactive and easy-to-use web interface for seamless predictions.

## Notes
- The model is based on historical dengue case data and may require periodic updates to maintain its accuracy.
- The prediction accuracy may vary based on the quality and relevance of the input data.