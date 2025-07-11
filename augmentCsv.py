import pandas as pd
import numpy as np
import itertools
import os

# ------------------------------
# 1. Load Data and Preprocess
# ------------------------------
file_path = "games_2022.csv"
df = pd.read_csv(file_path)
df.fillna(df.mean(numeric_only=True), inplace=True) #   Fill missing values with column means

# ------------------------------
# 2. Define Simulation & Scenario Settings
# ------------------------------

# Columns for Monte Carlo simulation (performance stats)
mc_columns = ['FGA_2', 'FGM_2', 'FGA_3', 'FGM_3', 'FTA', 'FTM',
              'AST', 'BLK', 'STL', 'TOV', 'TOV_team', 'DREB', 'OREB']

# Non-MC columns (use historical averages)
non_mc_columns = ['F_tech', 'F_personal', 'rest_days', 'attendance',
                  'tz_dif_H_E', 'prev_game_dist', 'travel_dist']

# Scenario parameters
num_rematches = 1             # synthetic games per pairing
fatigue_threshold_rest = 3    # if rest_days < this, then fatigue applies
fatigue_threshold_travel = 500  # if travel_dist > this, then fatigue applies
home_advantage_factor = 1.1   # multiplier for home team score
injury_prob = 0.1             # probability to apply injury adjustment
injury_factor = 0.85          # factor to reduce MC stats if injury applies

# Output file name
output_file = "aug_round_robin_montecarlo_scenario.csv"

# ------------------------------
# Real-Time CSV Appending Function
# ------------------------------
def append_to_csv(row):
    """Append a single row (dictionary) to the output CSV."""
    temp_df = pd.DataFrame([row])
    header = not os.path.exists(output_file) #Write header if file does not exist
    temp_df.to_csv(output_file, mode='a', index=False, header=header)

# Optionally remove the output file to start fresh.
if os.path.exists(output_file):
    os.remove(output_file)

# ------------------------------
# 3. Compute Per-Team Historical Statistics
# ------------------------------
teams = df['team'].unique()
team_stats = {}
for team in teams:
    team_data = df[df['team'] == team]
    # Compute mean and standard deviation for Monte Carlo columns
    mc_means = team_data[mc_columns].mean()
    mc_stds = team_data[mc_columns].std().fillna(0)
    # Compute mean for non-MC columns (if available)
    non_mc_means = {}
    for col in non_mc_columns:
        non_mc_means[col] = team_data[col].mean() if col in team_data.columns else 0
    # Store computed stats for the team
    team_stats[team] = {
        'mc_means': mc_means,
        'mc_stds': mc_stds,
        'non_mc': non_mc_means
    }

# ------------------------------
# 4. Define Simulation Functions
# ------------------------------
def simulate_mc_stats(team):
    """Simulate MC stats for a team based on historical means and stds."""
    stats = {}
    means = team_stats[team]['mc_means']
    stds = team_stats[team]['mc_stds']
    for col in mc_columns:
        m = means[col]
        s = stds[col]
        val = np.random.normal(m, s) if s > 0 else m  # Simulate stat using normal distribution
        stats[col] = max(0, int(round(val)))  # Ensure no negative values
    return stats

def get_non_mc_stats(team):
    """Return historical average values for non-MC columns (rounded)."""
    non_mc = team_stats[team]['non_mc']
    return {col: int(round(non_mc[col])) for col in non_mc}

def compute_team_score(mc_stats):
    """Compute team score as: 2*FGM_2 + 3*FGM_3 + FTM."""
    return mc_stats['FGM_2'] * 2 + mc_stats['FGM_3'] * 3 + mc_stats['FTM']

def simulate_overtime():
    """Simulate overtime minutes: 0 with probability 0.9, else random between 1 and 5."""
    return 0 if np.random.rand() < 0.9 else np.random.randint(1, 6)

# ------------------------------
# 5. Generate Synthetic Matches (Round Robin + Monte Carlo + Scenario Adjustments)
# ------------------------------
def simulate_team(team, is_home=False, apply_fatigue=False, apply_injury=False):
    mc_stats = simulate_mc_stats(team)
    non_mc_stats = get_non_mc_stats(team)

    # Apply fatigue adjustment for away teams if needed
    if apply_fatigue:
        if non_mc_stats.get('rest_days', 0) < fatigue_threshold_rest and non_mc_stats.get('travel_dist', 0) > fatigue_threshold_travel:
            for col in mc_columns:
                mc_stats[col] = int(round(mc_stats[col] * 0.9))

    # Apply injury adjustment with given probability
    if apply_injury and np.random.rand() < injury_prob:
        for col in mc_columns:
            mc_stats[col] = int(round(mc_stats[col] * injury_factor))

    # Compute team score using simulated MC stats
    score = compute_team_score(mc_stats)
    if is_home:
        score = int(round(score * home_advantage_factor)) # Apply home advantage

    mc_stats['team_score'] = score
    return mc_stats, non_mc_stats

# For each pairing of teams (round robin)
for teamA, teamB in itertools.combinations(teams, 2):
    for _ in range(num_rematches):
        # Randomly decide home and away
        if np.random.rand() < 0.5:
            home_team, away_team = teamA, teamB
        else:
            home_team, away_team = teamB, teamA

        # Generate common game_id and game_date
        game_id = f"synthetic_{np.random.randint(100000, 999999)}"
        game_date = pd.to_datetime(np.random.choice(df['game_date'])).strftime('%d-%m-%Y')

        # Simulate stats for both teams
        home_mc, home_non = simulate_team(home_team, is_home=True, apply_injury=True)
        away_mc, away_non = simulate_team(away_team, is_home=False, apply_fatigue=True, apply_injury=True)

        # Define team scores
        home_team_score = home_mc['team_score']
        away_team_score = away_mc['team_score']

        # Compute largest lead for the winning team (losing team gets 0)
        if home_team_score > away_team_score:
            largest_lead_home = home_team_score - away_team_score
            largest_lead_away = 0
        elif away_team_score > home_team_score:
            largest_lead_home = 0
            largest_lead_away = away_team_score - home_team_score
        else:
            largest_lead_home = largest_lead_away = 0

        # Simulate overtime minutes (same for both rows)
        OT_length_min_tot = simulate_overtime()
        notD1_incomplete = "FALSE"

        # Assemble home team row
        row_home = {
            "game_id": game_id,
            "game_date": game_date,
            "team": home_team,
            "FGA_2": home_mc.get("FGA_2", 0),
            "FGM_2": home_mc.get("FGM_2", 0),
            "FGA_3": home_mc.get("FGA_3", 0),
            "FGM_3": home_mc.get("FGM_3", 0),
            "FTA": home_mc.get("FTA", 0),
            "FTM": home_mc.get("FTM", 0),
            "AST": home_mc.get("AST", 0),
            "BLK": home_mc.get("BLK", 0),
            "STL": home_mc.get("STL", 0),
            "TOV": home_mc.get("TOV", 0),
            "TOV_team": home_mc.get("TOV_team", 0),
            "DREB": home_mc.get("DREB", 0),
            "OREB": home_mc.get("OREB", 0),
            "F_tech": home_non.get("F_tech", 0),
            "F_personal": home_non.get("F_personal", 0),
            "team_score": home_team_score,
            "opponent_team_score": away_team_score,
            "largest_lead": largest_lead_home,
            "notD1_incomplete": notD1_incomplete,
            "OT_length_min_tot": OT_length_min_tot,
            "rest_days": home_non.get("rest_days", 0),
            "attendance": home_non.get("attendance", 0),
            "tz_dif_H_E": home_non.get("tz_dif_H_E", 0),
            "prev_game_dist": home_non.get("prev_game_dist", 0),
            "home_away": "home",
            "home_away_NS": 1,
            "travel_dist": home_non.get("travel_dist", 0)
        }

        # Assemble away team row
        row_away = {
            "game_id": game_id,
            "game_date": game_date,
            "team": away_team,
            "FGA_2": away_mc.get("FGA_2", 0),
            "FGM_2": away_mc.get("FGM_2", 0),
            "FGA_3": away_mc.get("FGA_3", 0),
            "FGM_3": away_mc.get("FGM_3", 0),
            "FTA": away_mc.get("FTA", 0),
            "FTM": away_mc.get("FTM", 0),
            "AST": away_mc.get("AST", 0),
            "BLK": away_mc.get("BLK", 0),
            "STL": away_mc.get("STL", 0),
            "TOV": away_mc.get("TOV", 0),
            "TOV_team": away_mc.get("TOV_team", 0),
            "DREB": away_mc.get("DREB", 0),
            "OREB": away_mc.get("OREB", 0),
            "F_tech": away_non.get("F_tech", 0),
            "F_personal": away_non.get("F_personal", 0),
            "team_score": away_team_score,
            "opponent_team_score": home_team_score,
            "largest_lead": largest_lead_away,
            "notD1_incomplete": notD1_incomplete,
            "OT_length_min_tot": OT_length_min_tot,
            "rest_days": away_non.get("rest_days", 0),
            "attendance": away_non.get("attendance", 0),
            "tz_dif_H_E": away_non.get("tz_dif_H_E", 0),
            "prev_game_dist": away_non.get("prev_game_dist", 0),
            "home_away": "away",
            "home_away_NS": -1,
            "travel_dist": away_non.get("travel_dist", 0)
        }

        # Append both rows to the output CSV
        append_to_csv(row_home)
        append_to_csv(row_away)
        print(f"Appended match {game_id}: {home_team} (home) vs {away_team} (away)")

print(f"Synthetic augmented dataset saved as {output_file}")

# this code is for augmenting the existing csv, but according to requirement there is need
# of two more column derieved from this csv which done in ReqColumn.py