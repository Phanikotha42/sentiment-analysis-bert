import warnings
warnings.filterwarnings("ignore")  # we are skipping warnings to keep output clean

# importing requried libraries
import os
import csv
import re
import torch

# Fixing Streamlit's file watcher issue with PyTorch
torch.classes.__path__ = []

import pandas as pd
import streamlit as st
from streamlit.components.v1 import html
from transformers import BertTokenizerFast, BertForSequenceClassification

import pickle
from scipy.special import expit  # For SVM probability calculation
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import plotly.graph_objects as go

import networkx as nx  # graph structure
from pyvis.network import Network  # interactive visualization
from collections import Counter  # counting edges

import streamlit.components.v1 as components


#  DRAW PYVIS GRAPH 
def draw_pyvis_graph(G, height_px: int = 600):
    """
    here we are generating the interactive PyVis network from the NetworkX graph
    """
    net = Network(
        height=f"{height_px}px",
        width="100%",
        notebook=False,
        directed=True  # Arrows show mention direction
    )
    
    net.from_nx(G)  # Converting NetworkX to PyVis
    net.force_atlas_2based()  # applying force layout for nice spacing
    
    html_str = net.generate_html()
    components.html(html_str, height=height_px, scrolling=True)


# MODEL LOADING 
@st.cache_resource  # Caching models - loads only once
def load_models():
    """
    Loading all three models (LogReg, SVM, BERT) which are trained
    """
    # loading logistic regression model + vectorizer
    with open("logreg_model.pkl", "rb") as f:
        logreg_model = pickle.load(f)
    with open("logreg_vectorizer.pkl", "rb") as f:
        logreg_vectorizer = pickle.load(f)
    
    # loading svm classifier + vectorizer
    with open("svm_model.pkl", "rb") as f:
        svm_pipeline = pickle.load(f)
    svm_clf = svm_pipeline.steps[-1][1] # extracting classifier
    with open("svm_vectorizer.pkl", "rb") as f:
        svm_vectorizer = pickle.load(f)
    
    # Loading BERT tokenizer and classifier

    bert_tokenizer = BertTokenizerFast.from_pretrained("bert_model")
    bert_model = BertForSequenceClassification.from_pretrained("bert_model")
    bert_model.eval()  # Setting to evaluation mode
    
    return logreg_model, logreg_vectorizer, svm_vectorizer, svm_clf, bert_model, bert_tokenizer


# loading the models into memory
logreg_model, logreg_vectorizer, svm_vectorizer, svm_clf, bert_model, bert_tokenizer = load_models()


# here we are predicting the sentiment and returning label + confidence scores

def predict(text: str, model_choice: str):
   
    if model_choice == "Logistic Regression":
        X = logreg_vectorizer.transform([text])
        pred = logreg_model.predict(X)[0]
        probs = logreg_model.predict_proba(X)[0]


    elif model_choice == "SVM":
        X = svm_vectorizer.transform([text])
        pred = svm_clf.predict(X)[0]
        
        # SVM probability handling
        try:
            probs = svm_clf.predict_proba(X)[0]
        except AttributeError:
            # Using sigmoid for probability if predict_proba not available
            p = expit(svm_clf.decision_function(X)[0])
            probs = [1 - p, p]
            
    else:  # BERT
        # Tokenizing for BERT
        inputs = bert_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        
        # generating bert predictions
        with torch.no_grad():
            logits = bert_model(**inputs).logits
            probs_tensor = torch.nn.functional.softmax(logits, dim=1)[0]
        
        probs = probs_tensor.cpu().numpy()
        pred = int(probs_tensor.argmax())
    
    label = "Positive" if pred == 1 else "Negative"
    confidence = float(max(probs))
    
    return label, confidence, {"Negative": float(probs[0]), "Positive": float(probs[1])}


# path to log file
LOG_PATH = "tweet_logs.csv"

# Creating log file if missing
if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["timestamp", "model", "text", "prediction", "confidence"]
        )

# storing a prediction record with timestamp for analysis
def log_input(model: str, text: str, label: str, conf: float):
    
    ts = datetime.now(timezone.utc).isoformat()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([ts, model, text, label, f"{conf:.4f}"])


# creating the main Streamlit UI
st.title(" Tweet Analysis ")

# Creating two tabs sentiment + graph 

tabs = st.tabs(["Sentiment Classifier", "Knowledge Graph"])


# TAB 1: SENTIMENT CLASSIFIER 
with tabs[0]:
    st.sidebar.title("Settings")
    
    # Dark mode toggle
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False

    if st.sidebar.button(" Toggle Dark/Light", key="toggle_theme"):
        st.session_state.dark_mode = not st.session_state.dark_mode

    # Applying dark mode CSS 
    if st.session_state.dark_mode:
        st.markdown(
            """
            <style>
              .css-1d391kg, .css-12oz5g7 { background-color: #1e1e1e !important; color: #f1f1f1; }
              .stTextInput, .stSelectbox, .stButton button { background-color: #2e2e2e !important; color: white; }
            </style>
            """,
            unsafe_allow_html=True,
        )

    # Model selection
    model_choice = st.sidebar.selectbox(
        "Model", ["BERT", "Logistic Regression", "SVM"], key="model_choice"
    )
    
    # File uploader for batch processing
    csv_file = st.sidebar.file_uploader(
        "Upload CSV (no header, 6 cols)", type="csv", key="csv_uploader"
    )
    run_button = st.sidebar.button("Run predictions", key="run_batch")

    st.subheader(" Tweet Sentiment Classifier")

    # Batch predictions when CSV uploaded
    if csv_file and run_button:
        try:
            # Reading CSV with Twitter format
            df = pd.read_csv(
                csv_file,
                header=None,
                names=["label", "id", "date", "flag", "user", "text"],
                encoding="latin1",
            )
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

        # looping over each tweet 
        results = []
        for _, row in df.iterrows():
            text = str(row["text"])
            true_label = row["label"]
            
            pred_label, conf, probs = predict(text, model_choice)
            log_input(model_choice, text, pred_label, conf)
            
            results.append({
                "text": text,
                "true_label": true_label,
                "prediction": pred_label,
                "confidence": conf,
                "Neg_prob": probs["Negative"],
                "Pos_prob": probs["Positive"],
            })

        # Displaying results
        out_df = pd.DataFrame(results)
        st.success("Predictions complete!")
        st.dataframe(out_df[["text", "true_label", "prediction", "confidence"]])

        # Download button
        st.download_button(
            "Download results as CSV",
            data=out_df.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv",
            key="download_results"
        )

        # Converting labels to binary for metrics
        out_df["true_binary"] = out_df["true_label"].apply(lambda x: 1 if int(x) == 4 else 0)
        out_df["pred_binary"] = (out_df["prediction"] == "Positive").astype(int)
        
        y_true = out_df["true_binary"]
        y_pred = out_df["pred_binary"]
        y_score = out_df["Pos_prob"]

        # Calculating metrics
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", pos_label=1, zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        
        report_dict = classification_report(
            y_true, y_pred,
            target_names=["Negative", "Positive"],
            output_dict=True,
            zero_division=0
        )
        report_df = pd.DataFrame(report_dict).transpose()
        report_df["support"] = report_df["support"].astype(int)

        # Displaying metrics
        st.markdown("---")
        st.subheader("Overall Metrics")
        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Value": [f"{acc:.2%}", f"{prec:.2f}", f"{rec:.2f}", f"{f1:.2f}"]
        })
        st.table(metrics_df)

        # Confusion matrix
        st.subheader("Confusion Matrix")
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Negative", "Actual Positive"],
            columns=["Pred Negative", "Pred Positive"]
        )
        st.table(cm_df)

        # Classification report
        st.subheader("Classification Report")
        st.table(report_df)

        # ROC curve
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax.plot([0, 1], [0, 1], "k--")  # Diagonal reference line
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        st.pyplot(fig)

    # Single tweet prediction
    tweet = st.text_area("Enter tweet here:", key="single_tweet")
    
    if st.button("Predict", key="single_predict"):
        if not tweet.strip():
            st.warning("Please enter some text.")
        else:
            lbl, cf, probs = predict(tweet, model_choice)
            
            # Showing result with styling
            if lbl == "Positive":
                st.markdown("### ✅ Positive sentiment!")
                st.success("This tweet reads as positive.")
            else:
                st.markdown("### ❌ Negative sentiment.")
                st.error("This tweet reads as negative.")

            # Probability bar chart
            prob_df = pd.DataFrame.from_dict(
                probs, orient="index", columns=["probability"]
            )
            st.bar_chart(prob_df)
            st.caption(f"Confidence: **{cf:.2%}**")

            # Confidence gauge
            fig3 = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=cf * 100,
                    title={"text": "Confidence (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"thickness": 0.3},
                        "steps": [
                            {"range": [0, 50], "color": "lightgray"},
                            {"range": [50, 100], "color": "lightgreen"},
                        ],
                    },
                )
            )
            st.plotly_chart(fig3, width='stretch')
            log_input(model_choice, tweet, lbl, cf)

    # Footer
    st.markdown(
        "<hr><center> Built with Streamlit</center><hr>",
        unsafe_allow_html=True
    )


# ── TAB 2: INTERACTIVE USER & KEYWORD GRAPH ────────────────────────────────
with tabs[1]:
    st.subheader("🔍 Interactive Mention/Keyword Graph (PyVis)")
    
    DATA_PATH = "C:/Users/phani/OneDrive/Desktop/PDF's Sem-2/Practicum/Dataset.csv"

    # Checking if dataset exists
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found at {DATA_PATH}")
        st.stop()

    # Loading full dataset
    df_full = pd.read_csv(
        DATA_PATH,
        header=None,
        names=["label", "id", "date", "flag", "user", "text"],
        encoding="latin1",
    )

    # User inputs
    username = st.text_input("Filter by username (no @)", key="kg_user").strip().lower()
    keyword  = st.text_input("Filter by keyword", key="kg_keyword").strip().lower()
    
    # Slider for performance control
    max_rows = st.slider(
        "Max rows to consider",
        1000,
        min(200000, len(df_full)),
        20000,
        1000,
        key="kg_sample"
    )

    if not username and not keyword:
        st.info("Enter a username or keyword above to build the graph.")
    else:
        # Sampling dataset for performance
        df_sample = df_full.sample(min(len(df_full), max_rows), random_state=1)
        edges = Counter()

        # Building graph edges
        if username:
            # Mode: Show who this user mentions
            title = f"@{username} → mentions"
            mask_user = df_sample["user"].str.lower() == username
            
            for _, row in df_sample[mask_user].iterrows():
                for tgt in re.findall(r"@(\w+)", str(row["text"])):
                    edges[(username, tgt.lower())] += 1
        else:
            # Mode: Show who uses this keyword
            title = f"Keyword '{keyword}' → users"
            mask_key = df_sample["text"].str.lower().str.contains(keyword)
            
            for _, row in df_sample[mask_key].iterrows():
                edges[(keyword, row["user"].lower())] += 1

        # Getting top 50 connections
        top_edges = edges.most_common(50)
        
        if not top_edges:
            st.warning("No edges to display.")
        else:
            # Creating NetworkX graph
            G = nx.DiGraph()
            
            for (u, v), w in top_edges:
                G.add_edge(u, v, weight=w)

            st.markdown(f"**Graph:** {title}")
            
            # Drawing interactive graph
            draw_pyvis_graph(G, height_px=600)

            # Showing matching tweets
            if username:
                matched = df_sample[mask_user][["user", "text", "date"]].copy()
            else:
                matched = df_sample[mask_key][["user", "text", "date"]].copy()

            # Adding sentiment to matched tweets
            matched["Sentiment"] = matched["text"].apply(lambda t: predict(t, model_choice)[0])
            matched["Confidence"] = matched["text"].apply(lambda t: predict(t, model_choice)[1])

            st.markdown("### Matching Tweets with Sentiment")
            
            if matched.empty:
                st.info("No tweets matched your filter.")
            else:
                # Expandable section for tweets
                with st.expander(f"Show {len(matched)} raw tweets"):
                    st.dataframe(
                        matched.rename(columns={
                            "user": "User",
                            "text": "Tweet Text",
                            "date": "Date"
                        }),
                        height=300
                    )

