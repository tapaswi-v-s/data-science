# Predicting TTC Subway Delay

## Overview
This project focuses on predicting TTC (Toronto Transit Commission) subway delays using regression analysis. The dataset comprises various factors such as date, time, day of the week, station, delay code, delay time, gap between trains, direction of train, subway line, and train number. By analyzing these factors, I aim to develop a model that can predict subway delays, which can be invaluable for both the TTC and commuters.

## Dataset
The dataset used for this analysis contains the following columns:

1. **Date**: The date (YYYY/MM/DD) on which the delay occurred.
2. **Time**: The hour and minute of the day.
3. **Day**: The day of the week.
4. **Station**: The subway station name.
5. **Code**: The TTC delay code.
6. **Min Delay**: The delay time in minutes.
7. **Min Gap**: The time length (in minutes) between trains.
8. **Bound**: The direction of the train dependent on the line.
9. **Line**: TTC subway line (e.g., YU, BD, SHP, and SRT).
10. **Vehicle**: TTC train number.

## Tools Used
- Python
- Jupyter Notebook
- Libraries: 
  - Pandas
  - NumPy
  - Scikit-learn
  - Matplotlib
  - Seaborn
  - BeautifulSoup - For scrapping subway station names from Wikipedia

## Analysis Steps
1. **Data Preprocessing**: Cleaned and formatted the dataset for analysis.
2. **Exploratory Data Analysis (EDA)**: Explored relationships between variables and identified patterns.
3. **Feature Engineering**: Derived new features and encoded categorical variables.
4. **Model Building**: Utilized regression techniques to train predictive models.
5. **Model Evaluation**: Assessed model performance using relevant metrics.
6. **Deployment**: Deployed the final model for predicting TTC subway delays.

## Repository Structure
- `data/`: Contains the dataset used for analysis.
- `model/`: Saved model file.
- `TTC - Subway Delay Time Prediction.ipynb`: Jupyter notebooks detailing the analysis process.
- `README.md`: Overview of the project and instructions for replication.

## Usage
To replicate the analysis and predictions:

1. Clone this repository to your local machine.
2. Open the Jupyter notebook and follow the analysis steps.
3. For prediction, load the saved model and input new data for predictions.

## Contributions
Contributions to improve the analysis, model performance, or any related aspect are welcome. Fork this repository, make your changes, and submit a pull request.

## Credits
- Dataset Source: [TTC Subway Delay Data](https://www.toronto.ca/city-government/data-research-maps/open-data/open-data-catalogue/#a45bd45a-bc17-729d-eb9f-b57691291969)
- This project is developed by [Tapaswi Satyapanthi](https://www.linkedin.com/in/tapaswi-v-s/).

## License
This project is licensed under the [MIT License](LICENSE.txt).