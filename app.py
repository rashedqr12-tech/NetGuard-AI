import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="NetGuard AI",
    page_icon="🛡️"
)

st.title("🛡️ NetGuard AI")
st.subheader("Smart Network Failure Prediction System")

model = joblib.load("network_failure_model.pkl")

st.write("Enter the network metrics below:")

latency = st.number_input("Latency (ms)", min_value=0.0, value=50.0)
packet_loss = st.number_input("Packet Loss (%)", min_value=0.0, value=1.0)
bandwidth = st.number_input("Bandwidth (Mbps)", min_value=0.0, value=500.0)
cpu = st.number_input("CPU Usage (%)", min_value=0.0, max_value=100.0, value=50.0)
memory = st.number_input("Memory Usage (%)", min_value=0.0, max_value=100.0, value=50.0)
connections = st.number_input("Connections", min_value=0, value=150)

if st.button("Predict Network Status"):

    input_data = pd.DataFrame([[
        latency,
        packet_loss,
        bandwidth,
        cpu,
        memory,
        connections
    ]], columns=[
        "latency_ms",
        "packet_loss_pct",
        "bandwidth_mbps",
        "cpu_usage_pct",
        "memory_usage_pct",
        "connections"
    ])

    prediction = model.predict(input_data)[0]

    if prediction == "Normal":
        st.success("🟢 Network Status: Normal")

    elif prediction == "Warning":
        st.warning("🟡 Network Status: Warning")

    else:
        st.error("🔴 Network Status: Critical")
