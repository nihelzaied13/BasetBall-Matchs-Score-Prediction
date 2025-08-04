import http.client
import json
import csv
import pandas as pd
import joblib


# Function to fetch and save basketball standings
def fetch_and_save_basketball_standings(api_host, api_key, league, season):
    conn = http.client.HTTPSConnection(api_host)
    headers = {
        'x-rapidapi-host': api_host,
        'x-rapidapi-key': api_key
    }
    endpoint = f"/standings?league={league}&season={season}"
    print(f"Fetching standings from endpoint: {endpoint}")
    conn.request("GET", endpoint, headers=headers)
    res = conn.getresponse()
    data = res.read()
    print(f"API Response Status: {res.status}")
    response_json = json.loads(data.decode("utf-8"))
    print("Standings API Response:", response_json)
    output_file = f"nba_standings_{league}_{season.replace('-', '_')}.csv"

    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Position", "Team Name", "Played Games", "Wins", "Win Percentage",
            "Losses", "Loss Percentage", "Points For", "Points Against",
            "Form", "Description"
        ])
        if response_json.get("results", 0) > 0:
            standings = response_json["response"][0]
            for team in standings:
                writer.writerow([
                    team.get("position", ""),
                    team.get("team", {}).get("name", ""),
                    team.get("games", {}).get("played", ""),
                    team.get("games", {}).get("win", {}).get("total", ""),
                    team.get("games", {}).get("win", {}).get("percentage", ""),
                    team.get("games", {}).get("lose", {}).get("total", ""),
                    team.get("games", {}).get("lose", {}).get("percentage", ""),
                    team.get("points", {}).get("for", ""),
                    team.get("points", {}).get("against", ""),
                    team.get("form", ""),
                    team.get("description", "")
                ])
    print(f"Data successfully written to '{output_file}'")
    return output_file


# Function to fetch and save basketball games
def fetch_and_save_basketball_games(api_host, api_key, league, season):
    conn = http.client.HTTPSConnection(api_host)
    headers = {
        'x-rapidapi-host': api_host,
        'x-rapidapi-key': api_key
    }
    endpoint = f"/games?league={league}&season={season}"
    print(f"Fetching games from endpoint: {endpoint}")
    conn.request("GET", endpoint, headers=headers)
    res = conn.getresponse()
    data = res.read()
    print(f"API Response Status: {res.status}")
    decoded_data = json.loads(data.decode("utf-8"))
    print("Games API Response:", decoded_data)
    output_file = f"basketball_games_{league}_{season.replace('-', '_')}.csv"

    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Game ID', 'Date', 'Time', 'League Name', 'Season', 'Country',
            'Home Team', 'Away Team', 'Home Logo', 'Away Logo',
            'Home Q1 Score', 'Home Q2 Score', 'Home Q3 Score', 'Home Q4 Score', 'Home Total Score',
            'Away Q1 Score', 'Away Q2 Score', 'Away Q3 Score', 'Away Q4 Score', 'Away Total Score',
            'Game Status', 'Timezone'
        ])
        for game in decoded_data.get('response', []):
            writer.writerow([
                game.get('id', ""),
                game.get('date', ""),
                game.get('time', ""),
                game.get('league', {}).get('name', ""),
                game.get('league', {}).get('season', ""),
                game.get('country', {}).get('name', ""),
                game.get('teams', {}).get('home', {}).get('name', ""),
                game.get('teams', {}).get('away', {}).get('name', ""),
                game.get('teams', {}).get('home', {}).get('logo', ""),
                game.get('teams', {}).get('away', {}).get('logo', ""),
                game.get('scores', {}).get('home', {}).get('quarter_1', ""),
                game.get('scores', {}).get('home', {}).get('quarter_2', ""),
                game.get('scores', {}).get('home', {}).get('quarter_3', ""),
                game.get('scores', {}).get('home', {}).get('quarter_4', ""),
                game.get('scores', {}).get('home', {}).get('total', ""),
                game.get('scores', {}).get('away', {}).get('quarter_1', ""),
                game.get('scores', {}).get('away', {}).get('quarter_2', ""),
                game.get('scores', {}).get('away', {}).get('quarter_3', ""),
                game.get('scores', {}).get('away', {}).get('quarter_4', ""),
                game.get('scores', {}).get('away', {}).get('total', ""),
                game.get('status', {}).get('long', ""),
                game.get('timezone', "")
            ])
    print(f"Detailed data has been written to {output_file}")
    return output_file


# Function to fetch team statistics
def fetch_team_statistics(api_host, api_key, league, season):
    conn = http.client.HTTPSConnection(api_host)
    headers = {'x-rapidapi-host': api_host, 'x-rapidapi-key': api_key}
    endpoint = f"/games/statistics/teams?id={league}"
    conn.request("GET", endpoint, headers=headers)
    res = conn.getresponse()
    data = res.read()
    if res.status != 200:
        print(f"Error fetching team statistics: {res.status}")
        return pd.DataFrame()

    response_json = json.loads(data.decode("utf-8"))
    stats_list = response_json.get("response", [])

    stats_data = []
    for team_stats in stats_list:
        stats_data.append({
            "Team ID": team_stats.get("team", {}).get("id", ""),
            "Team Name": team_stats.get("team", {}).get("name", ""),
            "games.played.all": team_stats.get("games", {}).get("played", 0),
            "games.wins.all.percentage": team_stats.get("games", {}).get("win", {}).get("percentage", 0),
            "games.loses.all.percentage": team_stats.get("games", {}).get("lose", {}).get("percentage", 0),
            "points.for.average.all": team_stats.get("points", {}).get("for", 0) / max(1, team_stats.get("games", {}).get("played", 1)),
            "points.against.average.all": team_stats.get("points", {}).get("against", 0) / max(1, team_stats.get("games", {}).get("played", 1)),
            "games.wins.home.percentage": team_stats.get("games", {}).get("home", {}).get("win", {}).get("percentage", 0),
            "games.wins.away.percentage": team_stats.get("games", {}).get("away", {}).get("win", {}).get("percentage", 0),
            "points.for.average.home": team_stats.get("points", {}).get("home", {}).get("average", 0),
            "points.for.average.away": team_stats.get("points", {}).get("away", {}).get("average", 0),
            "points.against.average.home": team_stats.get("points", {}).get("home", {}).get("against", 0),
            "points.against.average.away": team_stats.get("points", {}).get("away", {}).get("against", 0),
        })

    stats_df = pd.DataFrame(stats_data)
    print("Fetched Team Statistics Data:")
    print(stats_df.head())
    return stats_df


# Function to merge basketball data
def merge_basketball_data(games_csv, standings_csv):
    games_df = pd.read_csv(games_csv)
    standings_df = pd.read_csv(standings_csv)

    print("Games DataFrame Columns:", games_df.columns)
    print("Standings DataFrame Columns:", standings_df.columns)

    standings_avg_df = standings_df.groupby('Team Name', as_index=False).agg({
        'Wins': 'mean',
        'Win Percentage': 'mean',
        'Losses': 'mean',
        'Loss Percentage': 'mean',
        'Points For': 'mean',
        'Points Against': 'mean'
    })

    merged_df = pd.merge(
        games_df,
        standings_avg_df,
        left_on='Home Team',
        right_on='Team Name',
        how='left',
        suffixes=('', '_Home')
    )

    home_columns = ['Wins', 'Win Percentage', 'Losses', 'Loss Percentage', 'Points For', 'Points Against']
    for col in home_columns:
        merged_df.rename(columns={col: f"Home_{col}"}, inplace=True)
    merged_df.drop(columns=['Team Name'], inplace=True)

    merged_df = pd.merge(
        merged_df,
        standings_avg_df,
        left_on='Away Team',
        right_on='Team Name',
        how='left',
        suffixes=('', '_Away')
    )
    for col in home_columns:
        merged_df.rename(columns={col: f"Away_{col}"}, inplace=True)
    merged_df.drop(columns=['Team Name'], inplace=True)

    return merged_df


# Fetch and merge data
def fetch_and_merge_basketball_data(api_host, api_key, league, season):
    games_csv = fetch_and_save_basketball_games(api_host, api_key, league, season)
    standings_csv = fetch_and_save_basketball_standings(api_host, api_key, league, season)

    try:
        return merge_basketball_data(games_csv, standings_csv)
    except Exception:
        stats_data = fetch_team_statistics(api_host, api_key, league, season)
        return stats_data


# Prepare prediction data
def prepare_prediction_data(league, api_host, api_key, fetch_and_merge_basketball_data):
    seasons = ['2024-2025' , '2024','2025']
    testdata = pd.DataFrame()

    for season in seasons:
        try:
            season_data = fetch_and_merge_basketball_data(api_host, api_key, league, season)
            if not season_data.empty:
                testdata = pd.concat([testdata, season_data], ignore_index=True)
        except Exception as e:
            print(f"Failed to fetch data for season {season}: {e}")

    if testdata.empty:
        print("No data fetched for any season.")
        return pd.DataFrame()

    testdata.to_csv('testdata_for_prediction.csv', index=False)
    testdata_for_prediction = testdata[testdata['Game Status'] == 'Not Started']
    return testdata_for_prediction


# Process input data
def process_data_main(input_data):
    list_home = input_data.get('Home Team', []).tolist() if 'Home Team' in input_data else []
    list_away = input_data.get('Away Team', []).tolist() if 'Away Team' in input_data else []
    data_without_teams = input_data.drop(columns=['Home Team', 'Away Team'], errors='ignore')
    return data_without_teams, list_home, list_away


# Filter relevant columns
def filter_relevant_columns_main(data):
    relevant_columns = [
        'Home_Wins', 'Home_Win Percentage', 'Home_Losses', 'Home_Loss Percentage',
        'Home_Points For', 'Home_Points Against',
        'Away_Wins', 'Away_Win Percentage', 'Away_Losses', 'Away_Loss Percentage',
        'Away_Points For', 'Away_Points Against'
    ]
    return data[relevant_columns]


# Create predictions DataFrame
def create_predictions_df(list_home, list_away, predicted_points, predicted_margin, predicted_win_probabilities):
    home_scores = predicted_points[:, 0]
    away_scores = predicted_points[:, 1]

    data = {
        'Home_team': list_home,
        'Away_team': list_away,
        'Home_score': home_scores,
        'Away_score': away_scores,
        'Margin_of_Victory': predicted_margin,
        'Probability_Winning_Home': predicted_win_probabilities
    }
    return pd.DataFrame(data)
