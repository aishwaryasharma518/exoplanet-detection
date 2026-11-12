# Exoplanet Detection System Using Machine Learning

A machine learning-based project that classifies stellar light curves to identify potential exoplanet transit signals. It uses a Random Forest Classifier to analyse extracted features from light-curve data and provides an interactive Streamlit interface for visualisation and prediction.

## Features

- Stellar light-curve visualisation.
- CSV file upload for light-curve analysis.
- Random Forest-based classification.
- Extraction of 11 statistical and transit-related features.
- Transit depth and minimum flux measurements.
- Sample planet and non-planet light-curve predictions.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Matplotlib
- Streamlit
- Joblib

## Machine Learning Approach

**Algorithm:** Random Forest Classifier

The project follows these steps:

1. Generate sample stellar light-curve data.
2. Add noise and simulate transit dips for planet-labelled samples.
3. Extract 11 features, including transit depth, signal-to-noise ratio, minimum flux, skewness, kurtosis and interquartile range.
4. Split the feature dataset into training and testing sets (80:20).
5. Train a Random Forest Classifier with 100 estimators.
6. Save the trained model using Joblib.
7. Use the model to classify light-curve data through a Streamlit application.

## Dataset

The current training script generates 20 synthetic stellar light curves:
- 10 planet-labelled samples.
- 10 non-planet-labelled samples.

Each sample contains time and normalized flux values.

**Note:** These are simulated samples, not verified observations downloaded from NASA's Kepler mission. No verified test accuracy is reported.

## Project Structure

```text
exoplanet-detection/
├── 09_exoplanet_UI.py
├── Complete_code.py
├── feature_matrix.csv
├── random_forest_model.pkl
├── requirements.txt
├── README.md
│
└── Star_data/
    ├── star_0_planet.csv
    ├── star_1_planet.csv
    ├── ...
    └── star_19_noplanet.csv
```

## How to Run

1. Clone the repository:

   `git clone https://github.com/aishwaryasharma518/exoplanet-detection.git`

2. Navigate to the project directory:

   `cd exoplanet-detection`

3. Install dependencies:

   `pip install -r requirements.txt`

4. Run the Streamlit application:

   `streamlit run 09_exoplanet_UI.py`

The trained model file `random_forest_model.pkl` and the `Star_data` folder must be present in the project directory.

## Author

**Aishwarya Sharma**

Final Year B.Tech Computer Science Engineering Student
