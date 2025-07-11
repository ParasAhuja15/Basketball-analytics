import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# List of input files to process
files = [
    "teamwise_stats_north.csv", 
    "teamwise_stats_south.csv", 
    "teamwise_stats_east.csv", 
    "teamwise_stats_west.csv"
]

# Loop through each file to process the data
for file in files:
    # Load the data for the current file
    df = pd.read_csv(file)

    # Create a more balanced ranking score
    df['Ranking_Score'] = df['Total_Wins'] * (df['Total_Wins'] / (df['Total_Matches'] + 1)) 

    # Select features
    features = ['Avg_AST', 'Avg_BLK', 'Avg_STL', 'Avg_TOV', 'Avg_TOV_team',
                'Avg_F_tech', 'Avg_F_personal', 'Avg_largest_lead', 
                'Avg_FGP_2', 'Avg_FGP_3', 'Avg_FT_Percentage', 'Avg_Total_Rebounds', 'Total_Matches']

    X = df[features]
    y = df['Ranking_Score']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train XGBoost Model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
    model.fit(X_train_scaled, y_train)

    # Predict ranking scores for the entire dataset
    df['Predicted_Ranking_Score'] = model.predict(scaler.transform(X))

    # Rank teams based on predicted scores
    df = df.sort_values(by='Predicted_Ranking_Score', ascending=False)
    df['Rank'] = range(1, len(df) + 1)

    # Save results for each region
    output_file = file.replace(".csv", "_ranking.csv")
    df.to_csv(output_file, index=False)
    print(f"Team ranking saved as {output_file}")
