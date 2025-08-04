🏀 Basketball Game Predictions
A lightweight project for predicting win probability, margin of victory, and team points for upcoming basketball games.
It includes:

A Streamlit web app (app.py) for interactive use

A CLI workflow (main.py)

Pretrained scikit-learn models stored in models/

Utility modules for fetching, preparing, and scoring data

✨ Features
Win Probability — logistic model returns home/away win probabilities

Margin & Points — regression models estimate margin of victory and expected team points

Data Fetching — integrates with API-Sports Basketball (RapidAPI) to pull fixtures/standings

One-Click CSV — export all predictions as a CSV from the app

🔧 Tech Stack
Python, scikit-learn, pandas, numpy, joblib

Streamlit (frontend)

API-Sports Basketball via RapidAPI

📁 Project Structure
bash
Copier
Modifier
Basketball_Project/
├─ app.py                       # Streamlit app
├─ main.py                      # CLI entry point
├─ classes.py                   # Model wrappers (win prob, points, margin)
├─ functions.py                 # Data fetch/merge/feature prep helpers
├─ models/
│  ├─ points_predictor_model.pkl
│  ├─ marginpredictor.pkl
│  ├─ win_model (1).pkl
│  └─ win_scaler.pkl
├─ requirements.txt
└─ Basketball_App_README.md     # (legacy notes)
🚀 Quickstart
1) Clone & setup
bash
Copier
Modifier
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>/Basketball_Project

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install streamlit  # streamlit is required by app.py
Note: requirements.txt includes core libs. If Streamlit isn’t present, install it as shown above.

2) Get API credentials
Create a RapidAPI key for API-Sports Basketball and note the host:

API Host: v1.basketball.api-sports.io

API Key: your personal RapidAPI key

🖥️ Run the Streamlit App
bash
Copier
Modifier
cd Basketball_Project
streamlit run app.py
Inside the app:

Enter League ID (numeric)

Enter your RapidAPI Key

(If prompted) confirm API Host: v1.basketball.api-sports.io

Click Predict to view probabilities, margins, and points

Use Download CSV to export results

🧑‍💻 Run via CLI
bash
Copier
Modifier
cd Basketball_Project
python main.py
You’ll be prompted for:

League ID

API Key (RapidAPI)

Predictions print to stdout and are saved to predictions.csv.

⚙️ Configuration Notes
Models: Pretrained .pkl files are loaded from Basketball_Project/models/ via joblib.

API Calls: functions.py uses HTTPS calls against the API-Sports Basketball endpoints to fetch standings/fixtures, then merges/engineers features for the models.

Feature Usage: The app detects whether extended team statistics are available and adapts features accordingly.

🧪 Models at a Glance
WinProbabilityEstimator (classification, scikit-learn Logistic Regression)

PointsPredictor (regression)

MarginPredictor (regression)
Models and any required scalers (e.g., win_scaler.pkl) are bundled under models/.

If you retrain models, update the corresponding .pkl files and keep file names consistent with app.py.

🛠️ Development
Lint/format as you prefer.

Keep secrets out of version control. For local dev, you can export an env var:

bash
Copier
Modifier
export RAPIDAPI_KEY="your_key_here"
and modify app.py/main.py to read from os.environ.get("RAPIDAPI_KEY") as a default.

❓ FAQ
Q: I get an authentication or 403 error.
A: Verify the API Key and that the Host is v1.basketball.api-sports.io. Also check your RapidAPI quota.

Q: The app shows “No data found for the given league.”
A: Confirm the league ID is valid for the season you’re querying and that fixtures/standings exist.

Q: Streamlit not found?
A: Run pip install streamlit.

📄 License
Add your preferred license (e.g., MIT) under LICENSE.

🤝 Contributing
Fork the repo

Create a feature branch: git checkout -b feat/your-feature

Commit changes: git commit -m "feat: add your feature"

Push and open a PR

📬 Contact
Open an issue for bugs/feature requests

Or reach out to the maintainers (add your contact here)
