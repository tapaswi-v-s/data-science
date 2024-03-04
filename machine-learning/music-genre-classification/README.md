# Music Genre Classification

## Overview
This project focuses on classifying the music genre of audio files using classification analysis. The dataset consists of audio files and features extracted from them, including various statistical and spectral features. By analyzing these features, I aim to develop models that can accurately classify music genres.

## Dataset
The dataset includes two parts:
1. Audio files of 10 genres.
2. Features extracted from the audio files, divided into two CSV files: `features_30_sec.csv` and `features_3_sec.csv`.

## Features
The CSV dataset contains the following columns:
- `filename`: Name of the audio file
- Various statistical and spectral features such as mean and variance of:
    - Chroma STFT
    - RMS
    - Spectral centroid
    - Spectral bandwidth
    - Zero crossing rate
    - Harmony
    - Perceptr
    - Tempo
    - Mel-frequency cepstral coefficients (MFCC)
    - Chroma frequencies
- `label`: Music genre label

## Exploratory Data Analysis (EDA)
- Explored single audio files of all 10 genres.
- Extracted and visualized various features including sound waves, spectrogram, tempo, and more.

## Preprocessing
Performed preprocessing tasks on the CSV dataset:
- Removed unwanted columns and extracted dependent variable `label`.
- Normalized the data using MinMaxScaler.

## Classification Models and Results
Performed classification using the following models:

1. **Logistic Regression**
    - Training Accuracy: 69.86%
    - Testing Accuracy: 67.77%

2. **K Nearest Neighbor**
    - Training Accuracy: 89.83%
    - Testing Accuracy: 85.05%

3. **Decision Tree**
    - Training Accuracy: 99.93%
    - Testing Accuracy: 64.06%

4. **Random Forest**
    - Training Accuracy: 96.71%
    - Testing Accuracy: 79.88%

5. **CatBoost**
    - Training Accuracy: 99.93%
    - Testing Accuracy: 89.62%

6. **XGBoost**
    - Training Accuracy: 99.93%
    - Testing Accuracy: 89.36%

XGBoost and CatBoost are the best-performing models with approximately 90% accuracy.

## Saved Models
The following models have been saved:
- `logistic_regression_model.pkl`
- `knn_model.pkl`
- `decision_tree_model.pkl`
- `random_forest_model.pkl`
- `catboost_model.cbm`
- `xgboost_model.xgb`

## Repository Structure
- `data/`: Contains the CSV files with extracted features and Raw Audio files of various music genres.
- `models/`: Contains the saved classification models.
- `Music-Genre-Classification.ipynb`: Jupyter notebooks detailing the exploratory data analysis, preprocessing steps, model building, and analysis process.
- `README.md`: Overview of the project and instructions for replication.

## Usage
To replicate the analysis and predictions:
1. Clone this repository to your local machine.
2. Open the Jupyter notebooks and follow the analysis steps.
3. For prediction, load the trained model and input new data for classification.

## Contributions
Contributions to improve the analysis, model performance, or any related aspect are welcome. Fork this repository, make your changes, and submit a pull request.

## Credits
- Dataset Source: [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data)
- This project was developed by [Tapaswi Satyapanthi](https://www.linkedin.com/in/tapaswi-v-s/).

## License
This project is licensed under the [MIT License](LICENSE.txt).

You are free to:
- Use the code for any purpose, including commercial purposes.
- Modify the code.
- Distribute the code.
- Sublicense the code.

Under the following terms:
- The code comes with no warranty or guarantee.
- You must include a copy of the license in any redistribution.
- You must provide appropriate credit to the original author (you).
