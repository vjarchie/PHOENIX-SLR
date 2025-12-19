# -*- coding: utf-8 -*-
"""
Training Dashboard for PHOENIX Sign Language Recognition

Real-time monitoring of training metrics including:
- Loss curves
- WER progress
- Blank ratio (CTC collapse indicator)
- GPU/System utilization
- Training speed

Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import torch
import subprocess
import re
import time
from pathlib import Path
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="PHOENIX SLR Training Dashboard",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #0f3460;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #e94560;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #a0a0a0;
    }
    .status-running {
        color: #00ff00;
        animation: pulse 2s infinite;
    }
    .status-stopped {
        color: #ff4444;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .stMetric {
        background-color: #1a1a2e;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #0f3460;
    }
</style>
""", unsafe_allow_html=True)

# Paths
CHECKPOINT_DIR = Path("checkpoints/transformer")
LOG_FILE = Path("training_log.jsonl")
TERMINAL_DIR = Path(r"c:\Users\admin\.cursor\projects\d-PHOENIX-SLR\terminals")


def get_gpu_stats():
    """Get GPU utilization stats using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw", 
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            values = result.stdout.strip().split(", ")
            return {
                "gpu_util": float(values[0]),
                "mem_used": float(values[1]),
                "mem_total": float(values[2]),
                "temperature": float(values[3]),
                "power": float(values[4]) if len(values) > 4 else 0
            }
    except Exception as e:
        pass
    return {"gpu_util": 0, "mem_used": 0, "mem_total": 16000, "temperature": 0, "power": 0}


def parse_terminal_output():
    """Parse the latest terminal output for training progress."""
    metrics = {
        "epoch": 0,
        "batch": 0,
        "total_batches": 2836,
        "loss": 0.0,
        "blank_ratio": 0.0,
        "speed": 0.0,
        "eta": "N/A",
        "is_running": False,
        "last_update": None
    }
    
    # Find the latest terminal file
    terminal_files = list(TERMINAL_DIR.glob("*.txt")) if TERMINAL_DIR.exists() else []
    if not terminal_files:
        return metrics
    
    latest_terminal = max(terminal_files, key=lambda x: x.stat().st_mtime)
    
    try:
        content = latest_terminal.read_text(encoding='utf-8', errors='ignore')
        lines = content.strip().split('\n')
        
        # Check if training is running
        metrics["is_running"] = "active_command" in content and "train.py" in content
        
        # Parse the last progress line
        for line in reversed(lines):
            # Match pattern like: Epoch 1:  44%|####4     | 1248/2836 [22:39<30:04,  1.14s/it, loss=6.2429, blank=100.0%]
            match = re.search(r'Epoch (\d+):\s+(\d+)%.*?(\d+)/(\d+).*?(\d+:\d+)<(\d+:\d+),\s+([\d.]+)(?:s/it|it/s).*?loss=([\d.]+).*?blank=([\d.]+)%', line)
            if match:
                metrics["epoch"] = int(match.group(1))
                metrics["batch"] = int(match.group(3))
                metrics["total_batches"] = int(match.group(4))
                speed_val = float(match.group(7))
                # Check if it's s/it or it/s
                if 's/it' in line:
                    metrics["speed"] = 1.0 / speed_val if speed_val > 0 else 0
                else:
                    metrics["speed"] = speed_val
                metrics["loss"] = float(match.group(8))
                metrics["blank_ratio"] = float(match.group(9))
                metrics["eta"] = match.group(6)
                metrics["last_update"] = datetime.now()
                break
            
            # Simpler match for progress
            simple_match = re.search(r'Epoch (\d+):', line)
            if simple_match and metrics["epoch"] == 0:
                metrics["epoch"] = int(simple_match.group(1))
                
    except Exception as e:
        st.error(f"Error parsing terminal: {e}")
    
    return metrics


def load_checkpoint_history():
    """Load training history from checkpoints."""
    history = []
    
    # Try to load from log file first
    if LOG_FILE.exists():
        try:
            with open(LOG_FILE, 'r') as f:
                for line in f:
                    history.append(json.loads(line))
        except:
            pass
    
    # Also check latest checkpoint
    latest_ckpt = CHECKPOINT_DIR / "latest.pth"
    if latest_ckpt.exists():
        try:
            ckpt = torch.load(latest_ckpt, map_location='cpu')
            if 'epoch' in ckpt:
                history.append({
                    'epoch': ckpt['epoch'] + 1,
                    'train_loss': ckpt.get('train_loss', 0),
                    'dev_loss': ckpt.get('dev_loss', 0),
                    'wer': ckpt.get('wer', 100),
                    'blank_ratio': ckpt.get('blank_ratio', 1.0)
                })
        except:
            pass
    
    return history


def load_model_config():
    """Load model configuration."""
    config_path = CHECKPOINT_DIR / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}


# Main Dashboard
def main():
    st.title("🤟 PHOENIX Sign Language Recognition")
    st.markdown("### Real-Time Training Dashboard")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        refresh_rate = st.slider("Refresh Rate (seconds)", 1, 30, 5)
        auto_refresh = st.checkbox("Auto Refresh", value=True)
        
        st.divider()
        
        st.header("📊 Model Info")
        config = load_model_config()
        if config:
            st.json(config)
        else:
            st.info("No model config found")
        
        st.divider()
        
        st.header("🎯 Target")
        st.metric("Target WER", "< 30%")
        st.metric("SOTA WER", "17.8%")
    
    # Auto refresh
    if auto_refresh:
        time.sleep(0.1)  # Small delay for UI
        st.empty()
    
    # Get current metrics
    terminal_metrics = parse_terminal_output()
    gpu_stats = get_gpu_stats()
    history = load_checkpoint_history()
    
    # Status indicator
    if terminal_metrics["is_running"]:
        st.markdown('<p class="status-running">● Training in Progress</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-stopped">● Training Stopped</p>', unsafe_allow_html=True)
    
    # Top metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Epoch",
            f"{terminal_metrics['epoch']}/100",
            delta=None
        )
    
    with col2:
        st.metric(
            "Current Loss",
            f"{terminal_metrics['loss']:.4f}",
            delta=f"{-0.5:.2f}" if terminal_metrics['loss'] > 0 else None
        )
    
    with col3:
        blank_color = "🔴" if terminal_metrics['blank_ratio'] > 80 else "🟡" if terminal_metrics['blank_ratio'] > 50 else "🟢"
        st.metric(
            f"Blank Ratio {blank_color}",
            f"{terminal_metrics['blank_ratio']:.1f}%"
        )
    
    with col4:
        st.metric(
            "Speed",
            f"{terminal_metrics['speed']:.2f} it/s"
        )
    
    with col5:
        st.metric(
            "ETA (Epoch)",
            terminal_metrics['eta']
        )
    
    st.divider()
    
    # Progress bars
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Epoch Progress")
        epoch_progress = terminal_metrics['batch'] / terminal_metrics['total_batches'] if terminal_metrics['total_batches'] > 0 else 0
        st.progress(epoch_progress, text=f"Batch {terminal_metrics['batch']}/{terminal_metrics['total_batches']}")
    
    with col2:
        st.subheader("🎯 Overall Progress")
        overall_progress = terminal_metrics['epoch'] / 100
        st.progress(overall_progress, text=f"Epoch {terminal_metrics['epoch']}/100")
    
    st.divider()
    
    # GPU Stats
    st.subheader("🖥️ GPU Utilization")
    gpu_col1, gpu_col2, gpu_col3, gpu_col4 = st.columns(4)
    
    with gpu_col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=gpu_stats['gpu_util'],
            title={'text': "GPU Compute"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#e94560"},
                'steps': [
                    {'range': [0, 30], 'color': "#1a1a2e"},
                    {'range': [30, 70], 'color': "#16213e"},
                    {'range': [70, 100], 'color': "#0f3460"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 2},
                    'thickness': 0.75,
                    'value': gpu_stats['gpu_util']
                }
            }
        ))
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    with gpu_col2:
        mem_percent = (gpu_stats['mem_used'] / gpu_stats['mem_total'] * 100) if gpu_stats['mem_total'] > 0 else 0
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=mem_percent,
            title={'text': "GPU Memory"},
            number={'suffix': '%'},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#00d4ff"},
                'steps': [
                    {'range': [0, 50], 'color': "#1a1a2e"},
                    {'range': [50, 80], 'color': "#16213e"},
                    {'range': [80, 100], 'color': "#0f3460"}
                ]
            }
        ))
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    with gpu_col3:
        st.metric("VRAM Used", f"{gpu_stats['mem_used']:.0f} MB")
        st.metric("VRAM Total", f"{gpu_stats['mem_total']:.0f} MB")
    
    with gpu_col4:
        st.metric("Temperature", f"{gpu_stats['temperature']:.0f}°C")
        st.metric("Power", f"{gpu_stats['power']:.0f} W")
    
    st.divider()
    
    # Training History Charts
    st.subheader("📊 Training History")
    
    if history:
        df = pd.DataFrame(history)
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Loss chart
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            if 'train_loss' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df['epoch'], y=df['train_loss'], name="Train Loss", 
                              line=dict(color='#e94560', width=2)),
                    secondary_y=False
                )
            if 'dev_loss' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df['epoch'], y=df['dev_loss'], name="Dev Loss",
                              line=dict(color='#00d4ff', width=2)),
                    secondary_y=False
                )
            
            fig.update_layout(
                title="Loss Curves",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                template="plotly_dark",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with chart_col2:
            # WER chart
            if 'wer' in df.columns:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(x=df['epoch'], y=df['wer'], name="WER",
                              line=dict(color='#00ff88', width=2),
                              fill='tozeroy', fillcolor='rgba(0,255,136,0.1)')
                )
                # Add target line
                fig.add_hline(y=30, line_dash="dash", line_color="yellow", 
                             annotation_text="Target: 30%")
                fig.add_hline(y=17.8, line_dash="dash", line_color="red",
                             annotation_text="SOTA: 17.8%")
                
                fig.update_layout(
                    title="Word Error Rate (WER)",
                    xaxis_title="Epoch",
                    yaxis_title="WER (%)",
                    template="plotly_dark",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("WER data will appear after first epoch completes")
    else:
        st.info("No training history yet. Metrics will appear after first epoch completes.")
    
    # Blank Ratio Monitor (CTC Collapse Detection)
    st.subheader("⚠️ CTC Collapse Monitor")
    
    blank_col1, blank_col2 = st.columns([2, 1])
    
    with blank_col1:
        blank_ratio = terminal_metrics['blank_ratio']
        
        # Color coding
        if blank_ratio > 80:
            color = "#ff4444"
            status = "DANGER - Possible CTC Collapse!"
            advice = "Consider: Lower learning rate, check data, increase training"
        elif blank_ratio > 50:
            color = "#ffaa00"
            status = "Warning - High blank ratio"
            advice = "Monitor closely, may improve with more training"
        else:
            color = "#00ff00"
            status = "Healthy - Model is learning"
            advice = "Training progressing normally"
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=blank_ratio,
            title={'text': f"Blank Token Ratio<br><span style='font-size:0.7em;color:{color}'>{status}</span>"},
            delta={'reference': 80, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 50], 'color': "rgba(0,255,0,0.2)"},
                    {'range': [50, 80], 'color': "rgba(255,170,0,0.2)"},
                    {'range': [80, 100], 'color': "rgba(255,0,0,0.2)"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        fig.update_layout(height=300, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    
    with blank_col2:
        st.markdown(f"""
        ### Status: {status}
        
        **Current Blank Ratio:** {blank_ratio:.1f}%
        
        **Advice:** {advice}
        
        ---
        
        **Thresholds:**
        - 🟢 < 50%: Healthy
        - 🟡 50-80%: Warning  
        - 🔴 > 80%: CTC Collapse
        """)
    
    st.divider()
    
    # Time Estimates
    st.subheader("⏱️ Time Estimates")
    
    time_col1, time_col2, time_col3 = st.columns(3)
    
    speed = terminal_metrics['speed']
    batches_remaining_epoch = terminal_metrics['total_batches'] - terminal_metrics['batch']
    epochs_remaining = 100 - terminal_metrics['epoch']
    
    if speed > 0:
        time_remaining_epoch = batches_remaining_epoch / speed / 60  # minutes
        time_remaining_total = (batches_remaining_epoch + epochs_remaining * terminal_metrics['total_batches']) / speed / 3600  # hours
        
        with time_col1:
            st.metric("Time to Complete Epoch", f"{time_remaining_epoch:.0f} min")
        
        with time_col2:
            st.metric("Total Time Remaining", f"{time_remaining_total:.1f} hours")
        
        with time_col3:
            completion_time = datetime.now() + timedelta(hours=time_remaining_total)
            st.metric("Estimated Completion", completion_time.strftime("%Y-%m-%d %H:%M"))
    else:
        st.info("Speed data not available yet")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>PHOENIX Sign Language Recognition - Training Dashboard</p>
        <p>Target: WER < 30% | SOTA: 17.8% (CorrNet, 2023)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()


if __name__ == "__main__":
    main()

