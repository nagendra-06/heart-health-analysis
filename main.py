import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#TRAIN ML MODEL ON SYNTHETIC DATA
def train_model():
    np.random.seed(42)
    n = 1000
    data = pd.DataFrame({
        "age":         np.random.randint(20, 80, n),
        "cholesterol": np.random.randint(150, 300, n),
        "bp":          np.random.randint(80, 200, n),
        "heart_rate":  np.random.randint(40, 130, n),
        "bmi":         np.round(np.random.uniform(16, 40, n), 1)
    })
    data["risk"] = (
        (data["age"] > 50) |
        (data["cholesterol"] > 240) |
        (data["bp"] > 150) |
        (data["heart_rate"] > 100) |
        (data["heart_rate"] < 50) |
        (data["bmi"] > 30)
    ).astype(int)

    X = data[["age", "cholesterol", "bp", "heart_rate", "bmi"]]
    y = data["risk"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"  ML Model Accuracy: {acc * 100:.1f}%")
    return model

#ODE — HEART RATE DYNAMICS OVER TIME
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
        "Bradycardia (low heart rate)":            "Do regular aerobic exercise to improve resting heart rate.",
        "Tachycardia (high heart rate)":           "Practice deep breathing and reduce caffeine intake.",
        "Hypertension (high blood pressure)":      "Reduce sodium, avoid smoking, exercise daily.",
        "Obesity-related cardiovascular risk":     "Adopt a calorie-controlled diet and increase activity.",
        "High cholesterol risk":                   "Reduce saturated fats, eat more fiber, exercise regularly."
    }
    return [tips[r] for r in risks if r in tips]

#HEALTH STATUS
def health_status(heart_rate, bmi, bp):
    if heart_rate < 50 or heart_rate > 100 or bp > 160:
        return "CRITICAL", "Consult a cardiologist immediately."
    elif bmi > 30 or bp > 140:
        return "AT RISK", "See a doctor for a check-up soon."
    elif 60 <= heart_rate <= 90 and bmi <= 25 and bp <= 120:
        return "HEALTHY", "Keep up your current lifestyle."
    else:
        return "MODERATE", "Monitor your health regularly."
        
#VISUALIZATIONS
def plot_results(sol, heart_rate, bmi, bp, cholesterol, age):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Heart Health Analysis", fontsize=14, fontweight="bold")

    #Plot 1 — Heart rate dynamics over time
    axes[0].plot(sol.t, sol.y[0], color="crimson", linewidth=2)
    axes[0].axhline(y=60,  color="green",  linestyle="--", alpha=0.6, label="Min normal (60)")
    axes[0].axhline(y=100, color="orange", linestyle="--", alpha=0.6, label="Max normal (100)")
    axes[0].set_title("Heart Rate Dynamics Over Time")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Heart Rate (bpm)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    #Plot 2 — Risk factor bar chart
    factors      = ["Heart Rate", "BMI", "Blood Pressure", "Cholesterol"]
    values       = [heart_rate, bmi, bp, cholesterol]
    normal_vals  = [75, 22, 120, 200]
    colors       = ["crimson" if v > n * 1.1 else "steelblue" for v, n in zip(values, normal_vals)]
    axes[1].bar(factors, values, color=colors)
    axes[1].set_title("Your Vitals (red = above normal)")
    axes[1].set_ylabel("Value")
    axes[1].grid(axis="y", alpha=0.3)
    for i, (v, n) in enumerate(zip(values, normal_vals)):
        axes[1].axhline(y=n, color="green", linestyle=":", alpha=0.4)

    #Plot 3 — Risk gauge (simple)
    risk_score = sum([
        heart_rate > 100 or heart_rate < 50,
        bmi > 30,
        bp > 140,
        cholesterol > 240,
        age > 50
    ])
    gauge_colors = ["green", "yellowgreen", "orange", "orangered", "crimson", "darkred"]
    axes[2].pie(
        [risk_score, 5 - risk_score],
        colors=[gauge_colors[risk_score], "#eeeeee"],
        startangle=90,
        wedgeprops={"width": 0.4}
    )
    axes[2].set_title(f"Risk Score: {risk_score}/5")
    axes[2].text(0, 0, f"{risk_score}/5", ha="center", va="center", fontsize=20, fontweight="bold")

    plt.tight_layout()
    plt.savefig("heart_report.png", dpi=150)
    plt.show()
    print("\n  Report saved as heart_report.png")

#MAIN
def main():
    print("=" * 50)
    print("       HEART HEALTH MONITOR")
    print("=" * 50)

    print("\nTraining ML model...")
    model = train_model()

    print("\nEnter your details:")
    heart_rate   = int(input("  Heart rate (bpm)       : "))
    bmi          = float(input("  BMI                    : "))
    bp           = float(input("  Blood pressure (mmHg)  : "))
    cholesterol  = int(input("  Cholesterol (mg/dL)    : "))
    age          = int(input("  Age                    : "))

    #ML prediction
    input_df = pd.DataFrame([[age, cholesterol, bp, heart_rate, bmi]],
                             columns=["age", "cholesterol", "bp", "heart_rate", "bmi"])
    ml_result = model.predict(input_df)[0]
    ml_prob   = model.predict_proba(input_df)[0][1] * 100

    #ODE simulation
    sol = solve_ivp(
        heart_rate_ode,
        (0, 10),
        [heart_rate],
        args=(heart_rate, bmi, bp, age),
        t_eval=np.linspace(0, 10, 200)
    )

    #Results
    status, advice = health_status(heart_rate, bmi, bp)
    risks = predict_diseases(heart_rate, bmi, bp, cholesterol)
    tips  = recovery_tips(risks)

    print("\n" + "=" * 50)
    print("           RESULTS")
    print("=" * 50)
    print(f"  Status         : {status}")
    print(f"  ML Risk Score  : {ml_prob:.1f}% cardiovascular risk")
    print(f"  Advice         : {advice}")

    if risks:
        print(f"\n  Detected Risks:")
        for r in risks:
            print(f"    • {r}")
    else:
        print("\n  Detected Risks  : None")

    if tips:
        print(f"\n  Recovery Tips:")
        for t in tips:
            print(f"    → {t}")
    else:
        print("\n  Tips: Keep maintaining your healthy lifestyle!")

    print("\nGenerating report...")
    plot_results(sol, heart_rate, bmi, bp, cholesterol, age)
    print("=" * 50)

if __name__ == "__main__":
    main()
