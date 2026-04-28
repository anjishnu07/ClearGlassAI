from flask import Flask, render_template, request, send_file, jsonify
import os
import pandas as pd
import plotly.express as px
import plotly.io as pio

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =====================================
# HOME PAGE
# =====================================
@app.route("/")
def home():
    return render_template("index.html")


# =====================================
# FILE UPLOAD + AUDIT PAGE
# =====================================
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "No file selected"

    file = request.files["file"]

    if file.filename == "":
        return "Please choose a file"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    if file.filename.endswith(".csv"):
        df = pd.read_csv(filepath)
    elif file.filename.endswith(".xlsx"):
        df = pd.read_excel(filepath)
    else:
        return "Only CSV and XLSX allowed"

    rows = df.shape[0]
    cols = df.shape[1]
    columns = df.columns.tolist()

    # Demo-safe fairness values
    accuracy = 0.8557
    mitigated_accuracy = 0.8298

    privileged_rate = 0.72
    unprivileged_rate = 0.34
    dir_score = 0.3402
    bias_status = "BIAS DETECTED"

    reweighed_dir = 0.7121
    exp_gradient_dir = 0.8012
    threshold_dir = 0.8441

    # Selection Rate Graph
    graph_data = pd.DataFrame({
        "Group": ["Privileged", "Unprivileged"],
        "Selection Rate": [72, 34]
    })

    fig = px.bar(
        graph_data,
        x="Group",
        y="Selection Rate",
        title="Selection Rate by Group",
        text="Selection Rate"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    selection_chart = pio.to_html(
        fig,
        full_html=False,
        config={"displayModeBar": False, "responsive": True}
    )

    # Confusion Matrix Graph
    confusion_data = pd.DataFrame({
        "Metric": ["TP", "FP", "FN", "TN"],
        "Value": [820, 312, 144, 2724]
    })

    fig2 = px.bar(
        confusion_data,
        x="Metric",
        y="Value",
        title="Model Confusion Matrix",
        text="Value"
    )

    fig2.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    confusion_chart = pio.to_html(
        fig2,
        full_html=False,
        config={"displayModeBar": False, "responsive": True}
    )

    # Feature Importance Graph
    shap_data = pd.DataFrame({
        "Feature": [
            "Age", "Income", "Education",
            "Hours per Week", "Experience", "Relationship"
        ],
        "Importance": [0.163, 0.151, 0.094, 0.083, 0.061, 0.052]
    })

    fig3 = px.bar(
        shap_data,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance",
        text="Importance"
    )

    fig3.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    shap_chart = pio.to_html(
        fig3,
        full_html=False,
        config={"displayModeBar": False, "responsive": True}
    )

    return render_template(
        "audit.html",
        filename=file.filename,
        rows=rows,
        cols=cols,
        columns=columns,
        accuracy=accuracy,
        mitigated_accuracy=mitigated_accuracy,
        privileged_rate=privileged_rate,
        unprivileged_rate=unprivileged_rate,
        dir_score=dir_score,
        bias_status=bias_status,
        reweighed_dir=reweighed_dir,
        exp_gradient_dir=exp_gradient_dir,
        threshold_dir=threshold_dir,
        selection_chart=selection_chart,
        confusion_chart=confusion_chart,
        shap_chart=shap_chart
    )


@app.route("/audit")
def audit():
    return render_template("audit.html")


# =====================================
# ANALYZE DATASET BUTTON API
# =====================================
@app.route("/analyze", methods=["POST"])
def analyze():
    result = {
        "dir": 0.3402,
        "spd": 0.1735,
        "eod": 0.0770,
        "bias_status": "BIAS DETECTED",
        "privileged_rate": 72,
        "unprivileged_rate": 34
    }
    return jsonify(result)


# =====================================
# EXPLAIN ROW BUTTON API
# =====================================
@app.route("/explain-row", methods=["POST"])
def explain_row():
    row_index = request.form.get("row_index", "0")

    result = {
        "row_index": row_index,
        "prediction": "Rejected",
        "main_reason": "Low experience and education mismatch",
        "bias_influence": "Sensitive feature influenced decision",
        "confidence": "82%"
    }

    return jsonify(result)


# =====================================
# MITIGATION PAGE
# =====================================
@app.route("/mitigation")
def mitigation():
    baseline_dir = 0.3402
    reweighed_dir = 0.7121
    exp_gradient_dir = 0.8012
    threshold_dir = 0.8441

    mitigation_data = pd.DataFrame({
        "Method": [
            "Baseline",
            "Reweighing",
            "Exp Gradient",
            "Threshold"
        ],
        "DIR": [
            baseline_dir,
            reweighed_dir,
            exp_gradient_dir,
            threshold_dir
        ]
    })

    fig = px.bar(
        mitigation_data,
        x="Method",
        y="DIR",
        title="Fairness Improvement After Bias Mitigation",
        text="DIR"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    mitigation_chart = pio.to_html(
        fig,
        full_html=False,
        config={"displayModeBar": False, "responsive": True}
    )

    return render_template(
        "mitigation.html",
        mitigation_chart=mitigation_chart,
        baseline_dir=baseline_dir,
        reweighed_dir=reweighed_dir,
        exp_gradient_dir=exp_gradient_dir,
        threshold_dir=threshold_dir
    )


# =====================================
# APPLY MITIGATION BUTTON API
# =====================================
@app.route("/apply-mitigation", methods=["POST"])
def apply_mitigation():
    result = {
        "baseline_dir": 0.3402,
        "mitigated_dir": 0.8441,
        "baseline_spd": 0.1735,
        "mitigated_spd": 0.0418,
        "baseline_eod": 0.0770,
        "mitigated_eod": 0.0281,
        "recommendation": "Threshold Optimization gives the best fairness improvement"
    }
    return jsonify(result)


# =====================================
# MULTI-PAGE PDF REPORT
# =====================================
@app.route("/download-report")
def download_report():
    return send_file("reports/fairness_report.pdf", as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
