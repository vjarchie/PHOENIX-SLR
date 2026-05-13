"""
CorrNet+ Training Dashboard
Real-time monitoring of training progress via Streamlit.
"""
import streamlit as st
import json
import time
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="CorrNet+ Training Dashboard",
    page_icon="🤟",
    layout="wide"
)

st.title("🤟 CorrNet+ Training Dashboard")
st.markdown("Real-time monitoring for PHOENIX-2014 Sign Language Recognition")

LOG_DIR = Path("logs")

def find_latest_run():
    """Find the most recent training run."""
    if not LOG_DIR.exists():
        return None
    
    runs = sorted(LOG_DIR.glob("run_*"), key=lambda x: x.stat().st_mtime, reverse=True)
    return runs[0] if runs else None

def load_training_log(run_dir: Path) -> pd.DataFrame:
    """Load training log from JSONL file."""
    log_file = run_dir / "training_log.jsonl"
    
    if not log_file.exists():
        return pd.DataFrame()
    
    records = []
    with open(log_file, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    return pd.DataFrame(records) if records else pd.DataFrame()

run_dir = find_latest_run()

if run_dir is None:
    st.warning("No training runs found. Start training first!")
    st.info("Run: `python train.py --config configs/phoenix2014.yaml`")
    
    if st.button("🔄 Refresh"):
        st.rerun()
else:
    st.sidebar.header("Training Run")
    st.sidebar.write(f"**Run**: {run_dir.name}")
    
    config_file = run_dir / "config.yaml"
    if config_file.exists():
        import yaml
        with open(config_file) as f:
            config = yaml.safe_load(f)
        
        st.sidebar.subheader("Configuration")
        st.sidebar.write(f"Backbone: {config['model']['backbone']}")
        st.sidebar.write(f"Hidden size: {config['model']['hidden_size']}")
        st.sidebar.write(f"Batch size: {config['training']['batch_size']}")
        st.sidebar.write(f"Accumulation: {config['training']['accumulation_steps']}")
        st.sidebar.write(f"LR: {config['training']['learning_rate']}")
    
    auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=True)
    
    df = load_training_log(run_dir)
    
    if df.empty:
        st.info("Waiting for training data...")
        if auto_refresh:
            time.sleep(5)
            st.rerun()
    else:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Epoch", f"{df['epoch'].iloc[-1]}")
        
        with col2:
            st.metric(
                "Train WER",
                f"{df['train_wer'].iloc[-1]:.2f}%",
                delta=f"{df['train_wer'].iloc[-1] - df['train_wer'].iloc[-2]:.2f}%" if len(df) > 1 else None,
                delta_color="inverse"
            )
        
        with col3:
            val_wer = df['val_wer'].iloc[-1] if df['val_wer'].iloc[-1] > 0 else df[df['val_wer'] > 0]['val_wer'].iloc[-1] if len(df[df['val_wer'] > 0]) > 0 else 0
            st.metric("Val WER", f"{val_wer:.2f}%")
        
        with col4:
            st.metric("Best WER", f"{df['best_wer'].iloc[-1]:.2f}%")
        
        st.markdown("---")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("📉 Word Error Rate")
            
            fig_wer = go.Figure()
            
            fig_wer.add_trace(go.Scatter(
                x=df['epoch'],
                y=df['train_wer'],
                mode='lines+markers',
                name='Train WER',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ))
            
            val_df = df[df['val_wer'] > 0]
            if not val_df.empty:
                fig_wer.add_trace(go.Scatter(
                    x=val_df['epoch'],
                    y=val_df['val_wer'],
                    mode='lines+markers',
                    name='Val WER',
                    line=dict(color='red', width=2),
                    marker=dict(size=6)
                ))
            
            fig_wer.add_hline(
                y=18.0,
                line_dash="dash",
                line_color="green",
                annotation_text="CorrNet+ SOTA (18%)"
            )
            
            fig_wer.add_hline(
                y=30.0,
                line_dash="dot",
                line_color="orange",
                annotation_text="Target (30%)"
            )
            
            fig_wer.update_layout(
                xaxis_title="Epoch",
                yaxis_title="WER (%)",
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                height=400
            )
            
            st.plotly_chart(fig_wer, use_container_width=True)
        
        with col_right:
            st.subheader("📊 Training Loss")
            
            fig_loss = go.Figure()
            
            fig_loss.add_trace(go.Scatter(
                x=df['epoch'],
                y=df['train_loss'],
                mode='lines+markers',
                name='Train Loss',
                line=dict(color='purple', width=2),
                marker=dict(size=4)
            ))
            
            val_loss_df = df[df['val_loss'] > 0]
            if not val_loss_df.empty:
                fig_loss.add_trace(go.Scatter(
                    x=val_loss_df['epoch'],
                    y=val_loss_df['val_loss'],
                    mode='lines+markers',
                    name='Val Loss',
                    line=dict(color='orange', width=2),
                    marker=dict(size=6)
                ))
            
            fig_loss.update_layout(
                xaxis_title="Epoch",
                yaxis_title="Loss",
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                height=400
            )
            
            st.plotly_chart(fig_loss, use_container_width=True)
        
        st.subheader("📈 Learning Rate Schedule")
        
        fig_lr = px.line(
            df,
            x='epoch',
            y='lr',
            title='Learning Rate',
            markers=True
        )
        fig_lr.update_layout(height=300)
        st.plotly_chart(fig_lr, use_container_width=True)
        
        st.subheader("📋 Training Log")
        st.dataframe(
            df[['epoch', 'lr', 'train_loss', 'train_wer', 'val_loss', 'val_wer', 'best_wer']].tail(20),
            use_container_width=True
        )
        
        if auto_refresh:
            time.sleep(5)
            st.rerun()
