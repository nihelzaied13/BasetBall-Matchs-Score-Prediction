from functions import (
    prepare_prediction_data,
    fetch_and_merge_basketball_data,
    process_data_main,
    filter_relevant_columns_main,
    create_predictions_df
)
from classes import WinProbabilityEstimator, PointsPredictor, MarginPredictor
import joblib


def main():
    # Step 1: Input parameters
    league = input("Enter the league ID: ")
    api_key = input("Enter your API key: ")
    api_host = "v1.basketball.api-sports.io"

    # Step 2: Fetch prediction data
    print("Fetching test data for prediction...")
    testdata_for_prediction = prepare_prediction_data(league, api_host, api_key, fetch_and_merge_basketball_data)

    if testdata_for_prediction.empty:
        print("No data available for prediction. Exiting...")
        return

    # Step 3: Check if team statistics are being used
    use_team_stats = "games.played.all" in testdata_for_prediction.columns

    # Step 4: Process the data
    print("Processing data for prediction...")
    if use_team_stats:
        print("Using team statistics for prediction.")
        data_test_clean = testdata_for_prediction  # Team stats already preprocessed
        list_home, list_away = None, None  # These are not needed for team statistics
    else:
        print("Using standings data for prediction.")
        data_test_no_teams, list_home, list_away = process_data_main(testdata_for_prediction)
        data_test_clean = filter_relevant_columns_main(data_test_no_teams)

    # Step 5: Load trained models and scalers
    print("Loading models...")
    points_predictor_model = joblib.load('points_predictor_model.pkl')
    win_model = joblib.load('win_model.pkl')
    win_scaler = joblib.load('win_scaler.pkl')
    margin_model = joblib.load('margin_model.pkl')

    # Step 6: Initialize model classes
    win_estimator = WinProbabilityEstimator(data=None)
    win_estimator.model = win_model
    win_estimator.scaler = win_scaler

    points_estimator = PointsPredictor(data=None)
    points_estimator.model = points_predictor_model

    margin_estimator = MarginPredictor(data=None)
    margin_estimator.model = margin_model

    # Step 7: Perform predictions
    print("Predicting win probabilities...")
    predicted_win_probabilities = win_estimator.predict_win_probability(data_test_clean, use_team_stats)

    print("Predicting margins...")
    predicted_margin = margin_estimator.predict(data_test_clean, use_team_stats)

    print("Predicting points...")
    predicted_points = points_estimator.predict(data_test_clean, use_team_stats)

    # Step 8: Combine results into a DataFrame
    print("Creating predictions DataFrame...")
    df_predictions = create_predictions_df(
        list_home if not use_team_stats else testdata_for_prediction["Team Name"].tolist(),
        list_away if not use_team_stats else testdata_for_prediction["Team Name"].tolist(),
        predicted_points,
        predicted_margin,
        predicted_win_probabilities
    )

    # Step 9: Output the predictions
    print("Predictions DataFrame:")
    print(df_predictions)

    # Save predictions to a CSV file (optional)
    df_predictions.to_csv('predictions.csv', index=False)
    print("Predictions saved to 'predictions.csv'.")


if __name__ == "__main__":
    main()
