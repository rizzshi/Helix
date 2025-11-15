"""Streamlit UI for Algorzen Helix — Interactive Forecast & Driver Analysis.
"""
import streamlit as st
import subprocess
import os
import sys
import pandas as pd
import json
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Algorzen Helix", page_icon="🧬", layout="wide")

# Header
st.title("🧬 Algorzen Helix")
st.markdown("**Predictive + Driver Intelligence Engine**")
st.markdown("*Algorzen Research Division © 2025 — Author: Rishi Singh*")
st.divider()

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Configuration")
    
    uploaded = st.file_uploader("📁 Upload CSV", type=["csv"], help="Upload your KPI history file")
    
    if uploaded:
        try:
            df_preview = pd.read_csv(uploaded)
            st.success(f"✓ Loaded {len(df_preview)} rows")
            uploaded.seek(0)  # Reset for later use
        except Exception as e:
            st.error(f"Error reading file: {e}")
    
    st.divider()
    
    kpi = st.text_input("📊 KPI Column", "revenue", help="Name of the column to forecast")
    horizon = st.number_input("📅 Forecast Horizon (days)", min_value=1, max_value=365, value=30)
    model = st.selectbox("🤖 Model", ["baseline", "gbm", "prophet"], 
                        help="baseline: 7-day MA, gbm: Gradient Boosting, prophet: Facebook Prophet")
    
    use_openai = st.checkbox("🤖 Use GPT-4 Narrative", help="Requires OPENAI_API_KEY env variable")
    
    st.divider()
    run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

# Main content
if run_button:
    input_path = "data/sample_kpi_history.csv"
    
    if uploaded is not None:
        input_path = os.path.join("data", "uploaded.csv")
        with open(input_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.info(f"Using uploaded file: {os.path.basename(input_path)}")
    else:
        st.info("Using sample dataset")
    
    output = os.path.join("reports", f"Helix_Streamlit_Report.pdf")
    
    cmd = ["python", "main.py", "--input", input_path, "--kpi", kpi, 
           "--horizon", str(horizon), "--model", model, "--output", output]
    
    if use_openai:
        cmd.append("--use-openai")
    
    with st.spinner("🔄 Running analysis pipeline..."):
        proc = subprocess.run(cmd, capture_output=True, text=True)
    
    # Show output
    with st.expander("📋 Pipeline Logs", expanded=False):
        st.code(proc.stdout if proc.stdout else proc.stderr)
    
    if os.path.exists(output):
        st.success("✅ Analysis Complete!")
        
        # Display metrics if available
        metadata_path = "reports/report_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Model", metadata.get("model_used", "N/A").upper())
            with col2:
                st.metric("KPI", metadata.get("kpi", "N/A").title())
            with col3:
                st.metric("Report ID", metadata.get("report_id", "N/A"))
            with col4:
                openai_status = "✓ Yes" if metadata.get("openai_used") else "✗ No"
                st.metric("GPT-4 Used", openai_status)
        
        st.divider()
        
        # Display visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Feature Importance")
            feat_imp_path = "reports/assets/feature_importance.png"
            if os.path.exists(feat_imp_path):
                st.image(feat_imp_path, use_container_width=True)
            else:
                st.info("Feature importance chart not generated")
        
        with col2:
            st.subheader("🔥 Correlation Heatmap")
            corr_path = "reports/assets/correlation_heatmap.png"
            if os.path.exists(corr_path):
                st.image(corr_path, use_container_width=True)
            else:
                st.info("Correlation heatmap not generated")
        
        st.divider()
        
        # Download button
        with open(output, "rb") as f:
            st.download_button(
                label="📥 Download Full PDF Report",
                data=f,
                file_name=os.path.basename(output),
                mime="application/pdf",
                use_container_width=True
            )
    else:
        st.error("❌ Report generation failed. Check logs above.")

else:
    # Landing page
    st.info("👈 Configure parameters in the sidebar and click **Run Analysis** to start")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 Multi-Model Forecasting")
        st.markdown("""
        - **Baseline**: 7-day moving average
        - **GBM**: Gradient Boosting (lag features)
        - **Prophet**: Seasonal + trend decomposition
        """)
    
    with col2:
        st.markdown("### 🔍 Driver Analysis")
        st.markdown("""
        - Feature importance ranking
        - SHAP explanations
        - Correlation analysis
        - Temporal precedence checks
        """)
    
    with col3:
        st.markdown("### 📈 Rich Metrics")
        st.markdown("""
        - MAE, RMSE, MAPE
        - R² Score
        - Directional accuracy
        - Confidence intervals
        """)
    
    st.divider()
    
    st.divider()
    st.caption("Algorzen Research Division © 2025 — Built with ❤️ by Rishi Singh")