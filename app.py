from flask import Flask, render_template, request
import numpy as np
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load model and scaler
model = joblib.load("best_xgb_model.pkl")
scaler = joblib.load("scaler.pkl")


# --- Start Page ---
@app.route("/")
def start():
    return render_template("start.html")


# --- Main Input Page ---
@app.route("/index")
def home():
    return render_template("index.html")


# --- Prediction Route ---
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # --- Step 1: Collect visible inputs from user ---
        subscribers = float(request.form.get("subscribers", 0))
        views = float(request.form.get("views", 0))
        video_count = float(request.form.get("video_count", 0))
        comment_count = float(request.form.get("comment_count", 0))
        account_age_years = float(request.form.get("account_age_years", 0))

        # --- Step 2: Compute ratios ---
        engagement_ratio = comment_count / views if views > 0 else 0
        view_to_subscriber_ratio = (
            views / subscribers if subscribers > 0 else 0
        )  # ✅ Corrected formula

        # --- Step 3: Prepare input features ---
        features = np.array(
            [
                [
                    subscribers,
                    views,
                    video_count,
                    comment_count,
                    account_age_years,
                    engagement_ratio,
                    view_to_subscriber_ratio,
                ]
            ]
        )

        # --- Step 4: Scale and predict ---
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]

        # --- Step 5: Label mapping ---
        popularity_dict = {0: "Low", 1: "Medium", 2: "High"}
        predicted_label = popularity_dict.get(prediction, "Unknown")

        # --- Step 6: Probability breakdown ---
        prob_dict = {}
        confidence = None
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(features_scaled)[0]
            confidence = round(np.max(probas) * 100, 2)
            prob_dict = {
                "Low": round(probas[0] * 100, 2),
                "Medium": round(probas[1] * 100, 2),
                "High": round(probas[2] * 100, 2),
            }
        else:
            confidence = 85.0
            prob_dict = {"Low": 20.0, "Medium": 30.0, "High": 50.0}

        # --- Step 7: Prepare data for display ---
        feature_names = [
            "Subscribers",
            "Views",
            "Video Count",
            "Comment Count",
            "Account Age (Years)",
            "Engagement Ratio (auto)",
            "View to Subscriber Ratio (auto)",
        ]

        feature_comparison = [
            {"Feature": name, "Value": val}
            for name, val in zip(feature_names, features[0])
        ]

        # --- Step 8: Render result page ---
        return render_template(
            "result.html",
            prediction=predicted_label,
            confidence=confidence,
            feature_comparison=feature_comparison,
            probability_breakdown=prob_dict,
        )

    except Exception as e:
        print("❌ Error:", e)
        return render_template(
            "result.html", prediction="Error: Invalid Input", confidence=0
        )


if __name__ == "__main__":
    app.run(debug=True)
