# Basketball-analytics

# Basketball Performance Analysis and Match Prediction

## Overview
This repository contains a comprehensive framework for analyzing and predicting the outcomes of basketball matches. The project leverages historical game data to perform statistical analysis, rank teams, and forecast future game results using a hybrid machine learning approach. The system combines team statistics, an XGBoost ranking model, a Random Forest classifier for win prediction, and an LSTM network for performance forecasting.

## Key Features

*   **Statistical Analysis & Feature Engineering:** The initial scripts process raw game data to calculate key performance indicators (KPIs) such as field goal percentages, free throw percentages, total rebounds, and assist-to-turnover ratios [1][2].
*   **Team Ranking System:** An XGBoost Regressor model is used to generate a data-driven ranking score for teams. This allows for a more nuanced understanding of team strength beyond simple win-loss records [3].
*   **Synthetic Data Generation:** To enhance the training dataset, a Monte Carlo simulation script generates synthetic round-robin match data. This process incorporates real-world variables like home-team advantage, team fatigue from travel, and potential player injuries [4].
*   **Time-Series Performance Forecasting:** An LSTM (Long Short-Term Memory) model analyzes sequences of past games to forecast a team's future statistical performance [2].
*   **Hybrid Match Prediction Model:** The core of the project is a hybrid prediction system. It combines a team's historical performance (both overall and head-to-head) with the LSTM-forecasted stats. This combined feature set is then fed into a Random Forest Classifier to predict the win probability for a given matchup [2].

## Repository Structure
```
.
├── data/
│   ├── games_2022.csv
│   ├── teamwise_stats.csv
│   └── teamwise_stats_ranking.csv
│
├── notebooks/
│   └── basketball_prediction.ipynb
│
├── models/
│   ├── random_forest_model.pkl
│   └── lstm_stat_predictor.h5
│
├── scripts/
│   ├── 1_data_preprocessing.py
│   ├── 2_team_ranking.py
│   ├── 3_augment_data.py
│   └── 4_add_opponent_column.py
│
├── requirements.txt
└── README.md
```

## Workflow
The project is structured as a sequential pipeline, from data ingestion to final prediction.

1.  **Data Preprocessing:** The `1_data_preprocessing.py` script cleans the raw `games_2022.csv` data, calculates derived statistics, and aggregates the results to create `teamwise_stats.csv` [1].
2.  **Team Ranking:** The `2_team_ranking.py` script applies an XGBoost model to the team statistics to generate a `Predicted_Ranking_Score` and ranks teams within each region [3].
3.  **Model Training & Prediction:** The `basketball_prediction.ipynb` notebook contains the end-to-end process for:
    *   Training the Random Forest classifier to predict game outcomes [2].
    *   Training the LSTM network to forecast future team stats [2].
    *   Integrating historical and forecasted stats to predict win probabilities for new matchups [2].
4.  **Data Augmentation (Optional):** The `3_augment_data.py` and `4_add_opponent_column.py` scripts can be used to generate a larger, synthetic dataset for model training [4][5].

## Getting Started

### Prerequisites
- Python 3.8+
- Pip (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone 
    cd 
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Execution

1.  **Prepare Team Statistics:**
    Run the data preprocessing script to generate aggregated team stats from the raw game logs.
    ```bash
    python scripts/1_data_preprocessing.py
    ```

2.  **Rank Teams:**
    Execute the ranking script to evaluate and rank teams based on the XGBoost model.
    ```bash
    python scripts/2_team_ranking.py
    ```

3.  **Train Models and Predict:**
    Open and run the `notebooks/basketball_prediction.ipynb` Jupyter Notebook. This will train the Random Forest and LSTM models, save them to the `/models` directory, and provide functions to predict outcomes for specified team matchups [2].

## Models Used
*   **XGBoost Regressor:** Utilized for creating a robust, feature-based team ranking system [3].
*   **Random Forest Classifier:** Serves as the primary model for classifying the binary outcome of a game (win or loss) [2].
*   **LSTM Network:** A deep learning model employed to capture temporal patterns in a team's performance and forecast future statistics [2].

## Dependencies
The project relies on the following major Python libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `tensorflow`
- `jupyter`
