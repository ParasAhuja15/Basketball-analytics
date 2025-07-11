import pandas as pd  # Import the pandas library for data manipulation

# Load the dataset containing game statistics
df = pd.read_csv(r"games_2022.csv")

# Drop unnecessary columns to reduce data size and focus on relevant statistics
df.drop(columns=['travel_dist', 'home_away_NS', 'home_away', 'attendance', 'game_date', 
                 'tz_dif_H_E', 'prev_game_dist', 'rest_days', 'OT_length_min_tot'], 
        inplace=True, errors='ignore')

# Load the dataset containing team-region mapping
regions = pd.read_csv(r"Team Region Groups.csv")

# Merge the main dataset with the region mapping based on the 'team' column
region_df = df.merge(regions, on="team", how="left")

# Fill missing region values with 'East' as the default region
region_df['region'] = region_df['region'].fillna('East')

# Calculate field goal percentage for 2-point shots
region_df['FGP_2'] = ((region_df['FGM_2'] / region_df['FGA_2']) * 100).replace([float('inf'), -float('inf')], 0).fillna(0)

# Calculate field goal percentage for 3-point shots
region_df['FGP_3'] = ((region_df['FGM_3'] / region_df['FGA_3']) * 100).replace([float('inf'), -float('inf')], 0).fillna(0)

# Calculate free throw percentage
region_df['FT_Percentage'] = ((region_df['FTM'] / region_df['FTA']) * 100).replace([float('inf'), -float('inf')], 0).fillna(0)

# Compute total rebounds by summing offensive and defensive rebounds
region_df['Total_Rebounds'] = region_df['DREB'] + region_df['OREB']

# Assign game result: 'True' for win, 'False' for loss
region_df['result'] = region_df.apply(lambda row: True if row['team_score'] > row['opponent_team_score'] 
                                      else False, axis=1)
region_df['result'] = region_df['result'].astype(bool)  # Ensure the result column is boolean

# Remove columns that are no longer needed after calculations
region_df.drop(columns=['FGA_2', 'FGA_3', 'FGM_2', 'FGM_3', 'FTA', 'FTM', 
                        'team_score', 'opponent_team_score', 'OREB', 'DREB', 
                        'notD1_incomplete'], inplace=True, errors='ignore')

# Group data by 'team' and calculate mean values for statistical features
teamwise_stats = region_df.groupby('team').agg({
    'AST': 'mean',  # Average assists
    'BLK': 'mean',  # Average blocks
    'STL': 'mean',  # Average steals
    'TOV': 'mean',  # Average turnovers
    'TOV_team': 'mean',  # Average team turnovers
    'F_tech': 'mean',  # Average technical fouls
    'F_personal': 'mean',  # Average personal fouls
    'largest_lead': 'mean',  # Average largest lead
    'FGP_2': 'mean',  # Average 2-point field goal percentage
    'FGP_3': 'mean',  # Average 3-point field goal percentage
    'FT_Percentage': 'mean',  # Average free throw percentage
    'Total_Rebounds': 'mean',  # Average total rebounds
    'result': ['sum', 'count'],  # Total wins and total matches played
    'region': 'first'  # Keep the region name (assuming one region per team)
})

# Rename columns for clarity
teamwise_stats.columns = ['Avg_AST', 'Avg_BLK', 'Avg_STL', 'Avg_TOV', 'Avg_TOV_team', 
                          'Avg_F_tech', 'Avg_F_personal', 'Avg_largest_lead', 
                          'Avg_FGP_2', 'Avg_FGP_3', 'Avg_FT_Percentage', 'Avg_Total_Rebounds', 
                          'Total_Wins', 'Total_Matches', 'Region']

# Save the aggregated team statistics to a CSV file
teamwise_stats.to_csv('teamwise_stats.csv')

# Define regions for filtering
regions = ['East', 'West', 'North', 'South']

# Split the dataset by region and save separate files
for region in regions:
    # Filter data for the specific region
    region_df = teamwise_stats[teamwise_stats['Region'] == region]
    
    # Save the region-specific statistics to a CSV file
    filename = f'teamwise_stats_{region.lower()}.csv'
    region_df.to_csv(filename, index=True)

    # Print confirmation message
    print(f"Saved: {filename}")