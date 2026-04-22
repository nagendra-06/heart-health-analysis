import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#TRAIN ML MODEL ON CSV DATA
def train_model():
    data = pd.read_csv("data.csv")  # <-- your file

    X = data[["age", "cholesterol", "bp", "heart_rate"]]
    y = data["risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"  ML Model Accuracy: {acc * 100:.1f}%")

    return model

#ODE — HEART RATE DYNAMICS
def heart_rate_ode(t, y, heart_rate, bmi, bp, age):
    stress  = 0.01 * (bmi - 22) + 0.02 * (bp - 120) + 0.03 * (heart_rate - 70)
    age_fac = 0.005 * (age - 35)
    dydt    = -0.1 * y + stress + age_fac
    return dydt

#DISEASE PREDICTION
def predict_diseases(heart_rate, bmi, bp, cholesterol):
    risks = []
    if heart_rate < 50:
        risks.append("Bradycardia (low heart rate)")
    if heart_rate > 100:
        risks.append("Tachycardia (high heart rate)")
    if bp > 140:
        risks.append("Hypertension (high blood pressure)")
    if bmi > 30:
        risks.append("Obesity-related cardiovascular risk")
    if cholesterol > 240:
        risks.append("High cholesterol risk")
    return risks

#RECOVERY TIPS
def recovery_tips(risks):
    tips = {
        "Bradycardia (low heart rate)": "Do regular aerobic exercise.",
        "Tachycardia (high heart rate)": "Reduce caffeine, practice breathing.",
        "Hypertension (high blood pressure)": "Reduce salt, exercise daily.",
        "Obesity-related cardiovascular risk": "Control diet, increase activity.",
        "High cholesterol risk": "Eat fiber, reduce fats."
    }
    return [tips[r] for r in risks if r in tips]

#HEALTH STATUS
def health_status(heart_rate, bmi, bp):
    if heart_rate < 50 or heart_rate > 100 or bp > 160:
        return "CRITICAL", "Consult a cardiologist immediately."
    elif bmi > 30 or bp > 140:
        return "AT RISK", "See a doctor soon."
    elif 60 <= heart_rate <= 90 and bmi <= 25 and bp <= 120:
        return "HEALTHY", "Keep up your lifestyle."
    else:
        return "MODERATE", "Monitor regularly."

#VISUALIZATION
def plot_results(sol, heart_rate, bmi, bp, cholesterol, age):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Heart Health Dashboard", fontsize=16, fontweight="bold")

    #HEART RATE DYNAMICS
    hr = sol.y[0]
    noise = np.random.normal(0, 1, len(hr))
    hr = hr + noise

    axes[0].plot(sol.t, hr, linewidth=2)
    axes[0].axhspan(60, 100, alpha=0.15)
    axes[0].axhline(60, linestyle="--", linewidth=1)
    axes[0].axhline(100, linestyle="--", linewidth=1)

    axes[0].set_title("Heart Rate Over Time")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("BPM")
    axes[0].grid(alpha=0.3)

    #VITALS BAR
    labels = ["HR", "BMI", "BP", "Chol"]
    values = [heart_rate, bmi, bp, cholesterol]

    colors = []
    for v, l in zip(values, labels):
        if (l == "HR" and (v < 60 or v > 100)) or \
           (l == "BMI" and v > 30) or \
           (l == "BP" and v > 140) or \
           (l == "Chol" and v > 240):
            colors.append("#ff4d4d")  # red
        else:
            colors.append("#4CAF50")  # green

    axes[1].bar(labels, values, color=colors)
    axes[1].set_title("Vitals Overview")
    axes[1].grid(axis="y", alpha=0.3)

    #RISK DONUT
    risk_score = sum([
        heart_rate > 100 or heart_rate < 50,
        bmi > 30,
        bp > 140,
        cholesterol > 240,
        age > 50
    ])

    colors = ["#ff4d4d", "#e0e0e0"]

    axes[2].pie(
        [risk_score, 5 - risk_score],
        colors=colors,
        startangle=90,
        wedgeprops={"width": 0.4}
    )

    axes[2].text(0, 0, f"{risk_score}/5",
                 ha="center", va="center",
                 fontsize=18, fontweight="bold")

    axes[2].set_title("Risk Score")

    plt.tight_layout()
    plt.savefig("heart_report.png", dpi=150)
    plt.show()

#MAIN
def main():
    print("=" * 50)
    print("       HEART HEALTH MONITOR")
    print("=" * 50)

    print("\nTraining ML model...")
    model = train_model()

    print("\nEnter your details:")
    heart_rate   = int(input("Heart rate (bpm): "))
    bmi          = float(input("BMI: "))
    bp           = float(input("Blood pressure: "))
    cholesterol  = int(input("Cholesterol: "))
    age          = int(input("Age: "))

    #ML Prediction
    input_df = pd.DataFrame([[age, cholesterol, bp, heart_rate]],
        columns=["age", "cholesterol", "bp", "heart_rate"])

    ml_prob = model.predict_proba(input_df)[0][1] * 100

    #ODE simulation
    sol = solve_ivp(
        heart_rate_ode,
        (0, 10),
        [heart_rate],
        args=(heart_rate, bmi, bp, age),
        t_eval=np.linspace(0, 10, 200)
    )

    status, advice = health_status(heart_rate, bmi, bp)
    risks = predict_diseases(heart_rate, bmi, bp, cholesterol)
    tips  = recovery_tips(risks)

    print("\nRESULTS")
    print(f"Status: {status}")
    print(f"ML Risk Score: {ml_prob:.1f}%")
    print(f"Advice: {advice}")

    if risks:
        print("\nRisks:")
        for r in risks:
            print(f" - {r}")

    if tips:
        print("\nTips:")
        for t in tips:
            print(f" - {t}")

    plot_results(sol, heart_rate, bmi, bp, cholesterol, age)


if __name__ == "__main__":
    main()
