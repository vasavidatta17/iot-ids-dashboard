streamlit
pandas
numpy
joblib
scikit-learn
xgboost
import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="IoT Intrusion Detection System", page_icon="🛡️", layout="wide")

REQUIRED = ["results/model_comparison.csv", "results/summary.json",
            "results/sample_test.csv", "models/edge_ids_model.pkl"]
missing = [f for f in REQUIRED if not os.path.exists(f)]
if missing:
    st.error(f"Missing files: {missing}. Run data1/comapare_models.py first.")
    st.stop()


@st.cache_data
def load_csv(path, **kw):
    return pd.read_csv(path, **kw)


@st.cache_resource
def load_bundle():
    return joblib.load("models/edge_ids_model.pkl")


def show(path, caption=None):
    if os.path.exists(path):
        st.image(path, caption=caption)
    else:
        st.warning(f"{path} not found. Run comapare_models.py again.")


def safe(n):
    return n.replace(" ", "_")


res = load_csv("results/model_comparison.csv")
summary = json.load(open("results/summary.json"))
bundle = load_bundle()
model, le, feats = bundle["model"], bundle["label_encoder"], bundle["features"]

# ---------- Header ----------
st.title("🛡️ AI Based Network Intrusion Detection System for IoT")
st.caption("Dataset: Edge-IIoTset | Models: XGBoost, Random Forest, Decision Tree, KNN")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Best Model", summary["best_model"])
c2.metric("Best Accuracy", f"{summary['best_accuracy']}%")
c3.metric("Records (cleaned)", f"{summary['samples']:,}")
c4.metric("Features", summary["features"])
c5.metric("Classes", summary["classes"])

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Model Comparison", "🧩 Confusion Matrix", "📈 Graphs", "🔎 Live Detection"])

# ---------- Tab 1 ----------
with tab1:
    st.subheader("Performance of all 4 algorithms")
    st.dataframe(res, hide_index=True)
    col_a, col_b = st.columns(2)
    with col_a:
        show("results/accuracy_comparison.png")
    with col_b:
        show("results/metrics_comparison.png")
    show("results/cv_comparison.png")

# ---------- Tab 2 ----------
with tab2:
    st.subheader("Confusion Matrix")
    choice = st.selectbox("Select model", list(res["Model"]))
    show(f"results/cm_{safe(choice)}.png", f"Confusion matrix - {choice}")

    st.markdown("**Best model, normalized (each row sums to 1)**")
    show("results/confusion_matrix_normalized.png")

    with st.expander("Show confusion matrix as a table (best model)"):
        if os.path.exists("results/confusion_matrix.csv"):
            st.dataframe(load_csv("results/confusion_matrix.csv", index_col=0))

# ---------- Tab 3 ----------
with tab3:
    st.subheader("Dataset and model insights")
    g1, g2 = st.columns(2)
    with g1:
        show("results/class_distribution.png")
    with g2:
        show("results/feature_importance.png")
    show("results/per_class_f1.png")
    with st.expander("Classification report (best model)"):
        if os.path.exists("results/classification_report.csv"):
            st.dataframe(load_csv("results/classification_report.csv", index_col=0))

# ---------- Tab 4 ----------
with tab4:
    st.subheader("Test the model on network records")
    sample = load_csv("results/sample_test.csv")
    options = ["Any (random)"] + sorted(sample["Actual"].unique())
    pick = st.selectbox("Attack type to test", options)
    n = st.slider("Number of records", 1, 20, 10)
    if st.button("Run detection"):
        pool = sample if pick == "Any (random)" else sample[sample["Actual"] == pick]
        batch = pool.sample(min(n, len(pool)))
        pred = le.inverse_transform(model.predict(batch[feats]))
        out = pd.DataFrame({"Actual": batch["Actual"].values, "Predicted": pred})
        out["Status"] = np.where(out["Predicted"] == "Normal", "🟢 Normal", "🔴 Attack")
        out["Result"] = np.where(out["Actual"] == out["Predicted"], "✅ Correct", "❌ Wrong")
        correct = int((out["Actual"] == out["Predicted"]).sum())
        m1, m2 = st.columns(2)
        m1.metric("Correctly detected", f"{correct} / {len(out)}")
        m2.metric("Attacks flagged", int((out["Predicted"] != "Normal").sum()))
        st.dataframe(out, hide_index=True)