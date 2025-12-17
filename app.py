import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Anomaly Detection System",
    layout="wide"
)

# ------------------ Title ------------------
st.title("Anomaly Detection in Sensor Data")
st.write(
    "Interactive machine learning system to detect abnormal sensor behavior "
    "in defence and industrial equipment."
)

# ------------------ File Upload ------------------
uploaded_file = st.file_uploader(
    "Upload Sensor CSV File",
    type=["csv"]
)

if uploaded_file is not None:
    # ------------------ Load Data ------------------
    df = pd.read_csv(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # ------------------ Feature Selection ------------------
    st.subheader("Sensor Selection")
    features = st.multiselect(
        "Select sensor columns for analysis",
        df.columns,
        default=list(df.columns)
    )

    if len(features) == 0:
        st.warning("Please select at least one sensor column.")
        st.stop()

    data = df[features]

    # ------------------ Sidebar Controls ------------------
    st.sidebar.header("Model Settings")

    contamination = st.sidebar.slider(
        "Anomaly Sensitivity",
        min_value=0.01,
        max_value=0.30,
        value=0.10,
        step=0.01,
        help="Higher value = more anomalies detected"
    )

    # ------------------ Scaling ------------------
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    st.success("Data scaled successfully")
    st.write("Scaled data shape:", scaled_data.shape)

    # ------------------ Isolation Forest Model ------------------
    model = IsolationForest(
        n_estimators=150,
        contamination=contamination,
        random_state=42
    )

    df["anomaly"] = model.fit_predict(scaled_data)
    anomalies = df[df["anomaly"] == -1]

    # ------------------ Metrics ------------------
    st.subheader("Anomaly Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Detected Anomalies", len(anomalies))
    col3.metric(
        "Anomaly Percentage",
        f"{(len(anomalies)/len(df))*100:.2f}%"
    )

    # ------------------ Visualization ------------------
    st.subheader("Anomaly Visualization")

    sensor_to_plot = st.selectbox(
        "Select sensor to visualize",
        features
    )

    plt.figure(figsize=(10, 4))
    plt.plot(
        df.index,
        df[sensor_to_plot],
        label="Sensor Reading",
        linewidth=2
    )
    plt.scatter(
        anomalies.index,
        anomalies[sensor_to_plot],
        color="red",
        s=60,
        label="Detected Anomaly"
    )

    plt.xlabel("Sample Index")
    plt.ylabel(sensor_to_plot)
    plt.title(f"Anomaly Detection on {sensor_to_plot}")
    plt.legend()
    plt.grid(alpha=0.3)

    st.pyplot(plt)

    # ------------------ Defence Context ------------------
    st.markdown("---")
    st.info(
        "This system applies unsupervised machine learning to monitor sensor "
        "data from defence equipment such as aircraft engines and radar systems. "
        "Early detection of abnormal behavior supports predictive maintenance "
        "and prevents critical system failures."
    )

else:
    st.info("Please upload a CSV file to start anomaly detection.")
