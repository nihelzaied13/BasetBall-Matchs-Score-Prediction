import streamlit as st
import joblib
import os
from classes import WinProbabilityEstimator, PointsPredictor, MarginPredictor
from functions import (
    prepare_prediction_data,
    fetch_and_merge_basketball_data,
    process_data_main,
    filter_relevant_columns_main,
    create_predictions_df
)

# Load models and scalers
base_path = os.path.dirname(__file__)  # Get the directory of the current script
models_path = os.path.join(base_path, 'models')
points_predictor_model = joblib.load(os.path.join(models_path, 'points_predictor_model.pkl'))
win_model = joblib.load(os.path.join(models_path, 'win_model (1).pkl'))
win_scaler = joblib.load(os.path.join(models_path, 'win_scaler.pkl'))
margin_model = joblib.load(os.path.join(models_path, 'marginpredictor.pkl'))


def main():
    st.title("Basketball Game Predictions")

    # Input fields for League ID and API Key
    league = st.text_input("Enter the League ID:")
    api_key = st.text_input("Enter your API Key:", type="password")  # Securely input API Key

    if st.button("Generate Predictions"):
        if league and api_key:
            st.write("Fetching all matches for the league...")

            # Fetch and prepare prediction data
            testdata_for_prediction = prepare_prediction_data(
                league,
                "v1.basketball.api-sports.io",
                api_key,
                fetch_and_merge_basketball_data
            )

            # Check if data was fetched successfully
            if testdata_for_prediction is None or len(testdata_for_prediction) == 0:
                st.error("No data found for the given league. Please check the League ID.")
                return

            # Display fetched matches
            st.write(f"Displaying all matches for league {league}:")
            st.dataframe(testdata_for_prediction)

            # Determine if team statistics are being used
            use_team_stats = "games.played.all" in testdata_for_prediction.columns

            # Process data for prediction
            st.write("Processing data for prediction...")
            if use_team_stats:
                st.write("Using team statistics for prediction.")
                data_test_clean = testdata_for_prediction  # Team stats already preprocessed
                list_home, list_away = None, None
            else:
                st.write("Using standings data for prediction.")
                data_test_no_teams, list_home, list_away = process_data_main(testdata_for_prediction)
                data_test_clean = filter_relevant_columns_main(data_test_no_teams)

            # Check if data is clean and ready for prediction
            if data_test_clean is None or len(data_test_clean) == 0:
                st.error("No valid data available for prediction. Please check the input data.")
                return

            # Predict Win Probabilities
            st.write("Predicting win probabilities...")
            win_estimator = WinProbabilityEstimator(data=None)
            win_estimator.model = win_model
            win_estimator.scaler = win_scaler
            predicted_win_probabilities = win_estimator.predict_win_probability(data_test_clean, use_team_stats)

            # Predict margins
            st.write("Predicting margins...")
            margin_estimator = MarginPredictor(data=None)
            margin_estimator.model = margin_model
            predicted_margin = margin_estimator.predict(data_test_clean, use_team_stats)

            # Predict team points
            st.write("Predicting points...")
            points_estimator = PointsPredictor(data=None)
            points_estimator.model = points_predictor_model
            predicted_points = points_estimator.predict(data_test_clean, use_team_stats)

            # Combine the results into a DataFrame
            st.write("Creating predictions DataFrame...")
            df_predictions = create_predictions_df(
                list_home if not use_team_stats else testdata_for_prediction["Team Name"].tolist(),
                list_away if not use_team_stats else testdata_for_prediction["Team Name"].tolist(),
                predicted_points,
                predicted_margin,
                predicted_win_probabilities
            )

            # Show the results in a table
            st.write("Predictions for all matches:")
            st.dataframe(df_predictions)

            # Provide an option to download predictions as CSV
            csv_file = df_predictions.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV",
                data=csv_file,
                file_name='predictions.csv',
                mime='text/csv'
            )

        else:
            st.error("Please provide both the League ID and your API Key.")


if __name__ == "__main__":
    main()
