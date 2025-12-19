# -*- coding: utf-8 -*-
"""
Speech-to-Sign Streamlit Demo

Interactive demo for converting text/speech to sign language video.

Run with:
    streamlit run demo_speech_to_sign.py
"""

import streamlit as st
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path
import json

# Page config
st.set_page_config(
    page_title="🤟 Speech to Sign",
    page_icon="🤟",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .gloss-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .gloss-token {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-weight: 500;
    }
    .arrow {
        color: #764ba2;
        font-size: 1.5rem;
        margin: 0 0.5rem;
    }
    .info-box {
        background: #e8f4f8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Load the Speech-to-Sign pipeline."""
    from src.speech_to_sign.pipeline import SpeechToSignPipeline
    from src.speech_to_sign.gloss_retriever import GlossVideoRetriever, build_default_index
    
    project_dir = Path(".")
    data_dir = project_dir / "data" / "phoenix2014-release"
    
    # Check if data exists
    if not data_dir.exists():
        return None, "PHOENIX data not found. Please download and extract the dataset."
    
    # Load vocab
    vocab_path = project_dir / "checkpoints" / "hybrid" / "vocab.json"
    if vocab_path.exists():
        with open(vocab_path, 'r', encoding='utf-8') as f:
            gloss_vocab = json.load(f)
    else:
        gloss_vocab = {}
    
    # Load or build index
    index_path = project_dir / "gloss_video_index.pkl"
    
    retriever = GlossVideoRetriever(str(data_dir))
    
    if index_path.exists():
        retriever.load_index(str(index_path))
    else:
        # Build index (first run)
        corpus_file = data_dir / "phoenix-2014-multisigner" / "annotations" / "manual" / "train.corpus.csv"
        if corpus_file.exists():
            retriever.build_index(
                str(corpus_file),
                save_path=str(index_path)
            )
    
    # Create pipeline
    pipeline = SpeechToSignPipeline.from_rule_based(str(project_dir), gloss_vocab)
    pipeline.gloss_retriever = retriever
    
    return pipeline, None


def display_glosses(glosses):
    """Display gloss sequence with nice formatting."""
    gloss_html = ""
    for i, gloss in enumerate(glosses):
        gloss_html += f'<span class="gloss-token">{gloss}</span>'
        if i < len(glosses) - 1:
            gloss_html += '<span class="arrow">→</span>'
    
    st.markdown(f'<div class="gloss-box">{gloss_html}</div>', unsafe_allow_html=True)


def save_video(frames, fps=25):
    """Save frames as video and return path."""
    import subprocess
    import shutil
    
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        temp_path = f.name
    
    height, width = frames.shape[1:3]
    
    # First save with mp4v
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    
    # Try to convert to H.264 using ffmpeg for browser compatibility
    if shutil.which('ffmpeg'):
        h264_path = temp_path.replace('.mp4', '_h264.mp4')
        try:
            subprocess.run([
                'ffmpeg', '-y', '-i', temp_path,
                '-c:v', 'libx264', '-preset', 'fast',
                '-pix_fmt', 'yuv420p', h264_path
            ], capture_output=True, check=True)
            os.unlink(temp_path)
            return h264_path
        except:
            pass
    
    return temp_path


def create_gif(frames, fps=10):
    """Create a GIF from frames (more browser compatible)."""
    from PIL import Image
    
    with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as f:
        gif_path = f.name
    
    # Convert to PIL images
    pil_frames = []
    for frame in frames[::2]:  # Skip every other frame for smaller GIF
        pil_frames.append(Image.fromarray(frame))
    
    if pil_frames:
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000 / fps),
            loop=0
        )
    
    return gif_path


def main():
    st.markdown('<h1 class="main-header">🤟 Speech to Sign Language</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <b>How it works:</b><br>
    1. Enter German text (or record speech)<br>
    2. Text is converted to DGS (German Sign Language) glosses<br>
    3. Sign language video is generated from gloss sequence
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Load pipeline
    with st.spinner("Loading pipeline..."):
        pipeline, error = load_pipeline()
    
    if error:
        st.error(f"❌ {error}")
        st.info("Please ensure the PHOENIX dataset is in `data/phoenix2014-release/`")
        return
    
    # Check if index is ready
    if len(pipeline.gloss_retriever.gloss_index) == 0:
        st.warning("⚠️ Video index not built yet. Building now (this may take a few minutes)...")
        return
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Input")
        
        input_mode = st.radio(
            "Input Mode",
            ["Text", "Speech"],
            horizontal=True
        )
        
        if input_mode == "Text":
            # Sample texts
            sample_texts = [
                "Morgen Regen Nord",
                "Heute Sonne warm",
                "Wochenende Schnee Alpen",
                "Wind stark Küste",
                "Temperatur zwanzig Grad"
            ]
            
            selected_sample = st.selectbox(
                "Choose a sample or enter custom text:",
                ["Custom..."] + sample_texts
            )
            
            if selected_sample == "Custom...":
                text_input = st.text_area(
                    "Enter German text:",
                    placeholder="E.g., Morgen gibt es Regen im Norden",
                    height=100
                )
            else:
                text_input = st.text_area(
                    "Enter German text:",
                    value=selected_sample,
                    height=100
                )
        
        else:  # Speech
            st.info("🎤 Speech input requires the Whisper package. Install with: `pip install openai-whisper`")
            
            try:
                audio_file = st.file_uploader(
                    "Upload audio file (WAV, MP3):",
                    type=['wav', 'mp3']
                )
                
                if audio_file:
                    st.audio(audio_file)
                    text_input = None  # Will be transcribed
                else:
                    text_input = None
            except Exception as e:
                st.error(f"Audio upload error: {e}")
                text_input = None
    
    with col2:
        st.subheader("⚙️ Options")
        
        clip_selection = st.selectbox(
            "Video clip selection:",
            ["random", "first", "longest"],
            help="How to select among multiple clips for each gloss"
        )
        
        show_debug = st.checkbox("Show debug info", value=False)
    
    st.markdown("---")
    
    # Generate button
    if st.button("🎬 Generate Sign Language Video", type="primary", use_container_width=True):
        if input_mode == "Text" and text_input:
            with st.spinner("Generating..."):
                try:
                    # Run pipeline
                    result = pipeline(text_input)
                    
                    # Display results
                    st.subheader("📋 Results")
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown("**Input Text:**")
                        st.write(result['text'])
                        
                        st.markdown("**Gloss Sequence:**")
                        display_glosses(result['glosses'])
                        
                        if show_debug:
                            st.markdown("**Debug Info:**")
                            st.json({
                                'num_glosses': len(result['glosses']),
                                'video_frames': result['video'].shape[0],
                                'video_size': f"{result['video'].shape[2]}x{result['video'].shape[1]}",
                                'duration_sec': result['video'].shape[0] / 25
                            })
                    
                    with col2:
                        st.markdown("**Sign Language Video:**")
                        
                        if result['video'].shape[0] > 0:
                            # Try GIF first (most compatible)
                            try:
                                gif_path = create_gif(result['video'], fps=12)
                                with open(gif_path, 'rb') as f:
                                    gif_bytes = f.read()
                                st.image(gif_bytes, caption="Sign Language Animation")
                                os.unlink(gif_path)
                            except Exception as e:
                                st.warning(f"GIF creation failed: {e}")
                            
                            # Also save MP4 for download
                            video_path = save_video(result['video'])
                            with open(video_path, 'rb') as f:
                                video_bytes = f.read()
                            
                            # Show video player
                            st.video(video_bytes)
                            
                            # Download button
                            st.download_button(
                                "⬇️ Download Video (MP4)",
                                video_bytes,
                                file_name="sign_language.mp4",
                                mime="video/mp4"
                            )
                            
                            # Cleanup
                            os.unlink(video_path)
                        else:
                            st.warning("No video frames generated. Some glosses may not be in the index.")
                
                except Exception as e:
                    st.error(f"Error generating video: {e}")
                    if show_debug:
                        import traceback
                        st.code(traceback.format_exc())
        
        elif input_mode == "Speech" and audio_file:
            st.warning("Speech transcription not yet implemented. Please use text input.")
        
        else:
            st.warning("Please enter some text or upload an audio file.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <b>PHOENIX Sign Language Production</b><br>
        Using PHOENIX-2014 dataset for German Sign Language (DGS)<br>
        <a href="https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/">Dataset Info</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

