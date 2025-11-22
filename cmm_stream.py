import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Content Monetization Modeler", layout="wide")
@st.cache_data(ttl=3600)
def load_data(path):
    df = pd.read_csv(path)
    return df

@st.cache_data(ttl=3600)
def prepare_dataset(df):
    df = df.copy()

    if 'video_id' in df.columns:
        df = df.drop(columns=['video_id'])
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['year'] = df['date'].dt.year.fillna(0).astype(int)
        df['month'] = df['date'].dt.month.fillna(0).astype(int)
        df['weekday'] = df['date'].dt.dayofweek.fillna(0).astype(int)
        df = df.drop(columns=['date'])
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
    df[num_cols] = df[num_cols].fillna(0)

    cat_cols = df.select_dtypes(include=['object']).columns
    for c in cat_cols:
        df[c] = df[c].fillna("unknown").astype(str)

    return df
@st.cache_resource
def train_model(df, target_col='ad_revenue_usd'):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
    X = df.drop(columns=[target_col])
    y = df[target_col].fillna(0)

    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()

    label_encoders = {}
    X_enc = X.copy()
    for c in cat_cols:
        le = LabelEncoder()
        X_enc[c] = le.fit_transform(X_enc[c].astype(str))
        label_encoders[c] = le

    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    metrics = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

    return {
        'model': model,
        'label_encoders': label_encoders,
        'numeric_cols': num_cols,
        'categorical_cols': cat_cols,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'metrics': metrics
    }

st.title("Content Monetization Modeler")
st.markdown("Predict `ad_revenue_usd` and explore basic analytics.")

st.sidebar.header("Data")
use_upload = st.sidebar.checkbox("Upload CSV file", value=False)

if use_upload:
    uploaded_file = st.sidebar.file_uploader("Upload cleaned_dataset.csv", type=["csv"])
    if uploaded_file is None:
        st.sidebar.warning("Please upload a CSV to continue.")
        st.stop()
    df_raw = pd.read_csv(uploaded_file)
else:
    default_path = r"C:\Guvi_Zen\content_monetization_modeler\youtube_ad_revenue_dataset.csv"
    st.sidebar.write(f"Using default file: `{default_path}`")
    try:
        df_raw = load_data(default_path)
    except FileNotFoundError:
        st.error(f"Default file not found at {default_path}. Upload a CSV instead.")
        st.stop()

st.sidebar.markdown("---")
st.sidebar.write("Tip: you can upload your own cleaned dataset with the same schema.")

with st.spinner("Preparing dataset..."):
    df = prepare_dataset(df_raw)

st.subheader("Dataset preview")
st.dataframe(df.head().T if st.checkbox("Show transposed preview", value=False) else df.head())

st.markdown(f"**Rows:** {df.shape[0]} — **Columns:** {df.shape[1]}")

if st.checkbox("Show summary statistics", value=True):
    st.subheader("Summary statistics (numerical columns)")
    st.dataframe(df.select_dtypes(include=['int64','float64']).describe().T)

with st.spinner("Training model (cached)..."):
    try:
        trained = train_model(df, target_col='ad_revenue_usd')
    except Exception as e:
        st.error(f"Model training failed: {e}")
        st.stop()

model = trained['model']
label_encoders = trained['label_encoders']
num_cols = trained['numeric_cols']
cat_cols = trained['categorical_cols']
metrics = trained['metrics']

st.subheader("Model evaluation (RandomForest)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("MAE", f"{metrics['MAE']:.4f}")
col2.metric("RMSE", f"{metrics['RMSE']:.4f}")
col3.metric("R²", f"{metrics['R2']:.4f}")
col4.metric("MSE", f"{metrics['MSE']:.4f}")

st.subheader("Visual analytics")

if st.checkbox("Show correlation heatmap (numerical)", value=True):
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

num_choice = st.selectbox("Select numeric column to visualize", options=num_cols)
if num_choice:
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4))
    sns.histplot(df[num_choice], bins=50, kde=True, ax=ax1)
    ax1.set_title(f"Histogram: {num_choice}")
    sns.boxplot(x=df[num_choice], ax=ax2)
    ax2.set_title(f"Boxplot: {num_choice}")
    st.pyplot(fig)

st.markdown("### Top categories (categorical columns)")
for c in cat_cols:
    top_n = 8
    top_vals = df[c].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(6,3))
    sns.barplot(x=top_vals.values, y=top_vals.index, ax=ax)
    ax.set_title(f"Top {top_n} in {c}")
    st.pyplot(fig)
if hasattr(model, "feature_importances_"):
    st.subheader("Model feature importance (RandomForest)")
    fi = pd.Series(model.feature_importances_, index=trained['X_train'].columns).sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x=fi.values, y=fi.index, ax=ax)
    ax.set_title("Top features by importance")
    st.pyplot(fig)

st.subheader("Actual vs Predicted (sample)")
X_test = trained['X_test']
y_test = trained['y_test']
y_pred = model.predict(X_test)
sample_df = pd.DataFrame({"actual": y_test, "predicted": y_pred}).sample(200, random_state=42)
fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(sample_df['actual'], sample_df['predicted'], alpha=0.6)
ax.set_xlabel("Actual ad_revenue_usd")
ax.set_ylabel("Predicted ad_revenue_usd")
ax.set_title("Actual vs Predicted (sample)")
st.pyplot(fig)

st.subheader("Predict ad_revenue_usd from user input")

st.write("Provide feature values below. For categorical fields select a value; numeric fields default to dataset medians.")

input_cols = trained['X_train'].columns.tolist()
with st.form("prediction_form"):
    user_input = {}
    for col in input_cols:
        if col in cat_cols:
        
            le = label_encoders[col]
            choices = list(le.classes_)
            
            if len(choices) > 50:
                choices = choices[:50]
            sel = st.selectbox(f"{col}", choices, key=f"inp_{col}")
            user_input[col] = int(le.transform([sel])[0])
        else:
            median_val = float(df[col].median()) if col in df.columns else 0.0
            val = st.number_input(f"{col}", value=median_val, format="%.6f", key=f"inp_num_{col}")
            user_input[col] = float(val)

    submitted = st.form_submit_button("Predict")
    if submitted:
        input_df = pd.DataFrame([user_input], columns=input_cols)
        pred_val = model.predict(input_df)[0]
        st.success(f"Predicted ad_revenue_usd: ${pred_val:.4f}")

st.markdown("---")
st.write("App built by:Kanagaraj k")
if st.button("Download sample prediction input CSV"):
    sample = pd.DataFrame([ {c: (df[c].median() if c in num_cols else (label_encoders[c].classes_[0] if c in label_encoders else "")) for c in input_cols} ])
    st.download_button("Download sample input", data=sample.to_csv(index=False), file_name="sample_input.csv", mime="text/csv")