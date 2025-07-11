import pandas as pd

# Load the CSV file
file_path = 'aug_round_robin_montecarlo_scenario.csv'
df = pd.read_csv(file_path)

# Create new columns for opponent team name and result
df['opponent_team'] = df['team'].iloc[::-1].values
df['result'] = (df['team_score'] > df['opponent_team_score']).astype(int)

# Save the updated dataframe as a new CSV
output_file = 'updated_aug_round_robin_montecarlo_scenario.csv'
df.to_csv(output_file, index=False)

df.head(), output_file
# this code is for adding two more column in augmented file.