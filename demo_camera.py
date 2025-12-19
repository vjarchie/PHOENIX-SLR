# -*- coding: utf-8 -*-
"""
Live Camera Demo for Sign Language Recognition.

Uses webcam to capture video and runs inference in real-time.

Usage:
    python demo_camera.py --checkpoint checkpoints/e2e/best.pth
"""

import sys
import argparse
import time
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models.transformer import HybridCTCAttentionModel
from translation.gloss_to_english import translate_glosses


class SignLanguageDemo:
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load checkpoint
        print(f"Loading model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.vocab = checkpoint['vocab']
        self.idx2gloss = {v: k for k, v in self.vocab.items()}
        
        # Special tokens
        self.sos_idx = self.vocab['<sos>']
        self.eos_idx = self.vocab['<eos>']
        self.pad_idx = self.vocab['<pad>']
        self.blank_idx = self.vocab['<blank>']
        
        # Initialize model
        self.model = HybridCTCAttentionModel(
            vocab_size=len(self.vocab),
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=3,
            dim_feedforward=2048,
            dropout=0.2,
            max_seq_len=500,
            ctc_weight=0.3,
            use_resnet=True
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded! Vocabulary: {len(self.vocab)} glosses")
        
        # Frame buffer for temporal context
        self.frame_buffer = deque(maxlen=64)
        self.last_prediction = ""
        self.last_translation = ""
        self.prediction_history = deque(maxlen=5)
        
    def preprocess_frame(self, frame):
        """Preprocess a single frame."""
        # Resize to PHOENIX dimensions (210x260)
        frame = cv2.resize(frame, (210, 260))
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        return frame
    
    def predict(self):
        """Run prediction on buffered frames."""
        if len(self.frame_buffer) < 16:
            return None
        
        # Stack frames: (T, H, W, C) -> (T, C, H, W)
        frames = np.array(list(self.frame_buffer))
        frames = torch.from_numpy(frames).float()
        frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W)
        
        # Add batch dimension
        frames = frames.unsqueeze(0).to(self.device)  # (1, T, C, H, W)
        
        with torch.no_grad():
            # Greedy decode
            pred_tokens = self.model.greedy_decode(
                frames,
                src_key_padding_mask=None,
                max_len=30,
                sos_idx=self.sos_idx,
                eos_idx=self.eos_idx
            )
            
            # Convert to glosses
            pred_seq = []
            for token_id in pred_tokens[0].tolist():
                if token_id == self.eos_idx:
                    break
                if token_id not in [self.sos_idx, self.pad_idx, self.blank_idx]:
                    if token_id in self.idx2gloss:
                        gloss = self.idx2gloss[token_id]
                        if not gloss.startswith('__') and not gloss.startswith('<'):
                            pred_seq.append(gloss)
            
            # Get English translation
            if pred_seq:
                english = translate_glosses(pred_seq)
                return ' '.join(pred_seq), english
            return "", ""
    
    def run(self, camera_id: int = 0):
        """Run live camera demo."""
        print("\n" + "=" * 60)
        print("Sign Language Recognition - Live Demo")
        print("=" * 60)
        print("Controls:")
        print("  [SPACE] - Capture and predict")
        print("  [C]     - Clear buffer")
        print("  [R]     - Record mode (continuous)")
        print("  [Q]     - Quit")
        print("=" * 60 + "\n")
        
        # Try different backends for Windows compatibility
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        cap = None
        
        for backend in backends:
            print(f"Trying camera {camera_id} with backend {backend}...")
            cap = cv2.VideoCapture(camera_id, backend)
            if cap.isOpened():
                # Test read a frame
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print(f"Camera opened successfully with backend {backend}")
                    break
                cap.release()
            cap = None
        
        if cap is None:
            print(f"Error: Could not open camera {camera_id} with any backend")
            print("Try: python demo_camera.py --camera 1")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 25)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for real-time
        
        recording = False
        last_predict_time = 0
        predict_interval = 2.0  # Predict every 2 seconds in record mode
        
        print("Camera opened! Press SPACE to capture, Q to quit.")
        
        frame_errors = 0
        max_frame_errors = 30  # Allow some frame drops
        
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                frame_errors += 1
                if frame_errors > max_frame_errors:
                    print(f"Error: Too many frame read failures ({frame_errors})")
                    break
                time.sleep(0.01)  # Brief pause before retry
                continue
            
            frame_errors = 0  # Reset on success
            
            # Display frame
            display_frame = frame.copy()
            
            # Add overlay info
            h, w = display_frame.shape[:2]
            
            # Status bar
            cv2.rectangle(display_frame, (0, 0), (w, 60), (0, 0, 0), -1)
            
            # Recording indicator
            if recording:
                cv2.circle(display_frame, (30, 30), 15, (0, 0, 255), -1)
                cv2.putText(display_frame, "RECORDING", (55, 38), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Auto-capture frames
                processed = self.preprocess_frame(frame)
                self.frame_buffer.append(processed)
                
                # Auto-predict periodically
                if time.time() - last_predict_time > predict_interval:
                    result = self.predict()
                    if result:
                        prediction, translation = result
                        if prediction:
                            self.last_prediction = prediction
                            self.last_translation = translation
                            self.prediction_history.append(prediction)
                    last_predict_time = time.time()
            else:
                cv2.putText(display_frame, f"Buffer: {len(self.frame_buffer)}/64 frames", 
                           (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Prediction display (larger area for both German and English)
            cv2.rectangle(display_frame, (0, h-100), (w, h), (0, 0, 0), -1)
            
            # German glosses
            cv2.putText(display_frame, "German (DGS):", (10, h-75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            if self.last_prediction:
                pred_text = self.last_prediction[:45] + "..." if len(self.last_prediction) > 45 else self.last_prediction
                cv2.putText(display_frame, pred_text, (120, h-75), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # English translation
            cv2.putText(display_frame, "English:", (10, h-35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            if self.last_translation:
                trans_text = self.last_translation[:50] + "..." if len(self.last_translation) > 50 else self.last_translation
                cv2.putText(display_frame, trans_text, (120, h-35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show ROI guide (where to sign)
            roi_x, roi_y = w//4, h//6
            roi_w, roi_h = w//2, 2*h//3
            cv2.rectangle(display_frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), 
                         (0, 255, 255), 2)
            cv2.putText(display_frame, "Sign here", (roi_x + roi_w//3, roi_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow('Sign Language Recognition', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord(' '):  # SPACE - capture frame
                processed = self.preprocess_frame(frame)
                self.frame_buffer.append(processed)
                print(f"Frame captured. Buffer: {len(self.frame_buffer)}/64")
                
                # Auto-predict when buffer is full enough
                if len(self.frame_buffer) >= 32:
                    result = self.predict()
                    if result:
                        prediction, translation = result
                        if prediction:
                            self.last_prediction = prediction
                            self.last_translation = translation
                            print(f"German: {prediction}")
                            print(f"English: {translation}")
            elif key == ord('c'):  # C - clear buffer
                self.frame_buffer.clear()
                self.last_prediction = ""
                self.last_translation = ""
                print("Buffer cleared")
            elif key == ord('r'):  # R - toggle recording
                recording = not recording
                if recording:
                    self.frame_buffer.clear()
                    self.last_prediction = ""
                    self.last_translation = ""
                    last_predict_time = time.time()
                    print("Recording started - perform sign language")
                else:
                    # Final prediction when stopping
                    result = self.predict()
                    if result:
                        prediction, translation = result
                        if prediction:
                            self.last_prediction = prediction
                            self.last_translation = translation
                            print(f"German: {prediction}")
                            print(f"English: {translation}")
                    print("Recording stopped")
            elif key == ord('p'):  # P - force predict
                result = self.predict()
                if result:
                    prediction, translation = result
                    if prediction:
                        self.last_prediction = prediction
                        self.last_translation = translation
                        print(f"German: {prediction}")
                        print(f"English: {translation}")
                    else:
                        print("No prediction generated")
                else:
                    print("Not enough frames to predict")
        
        cap.release()
        cv2.destroyAllWindows()
        print("\nDemo ended.")


def main():
    parser = argparse.ArgumentParser(description='Live Sign Language Recognition Demo')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/e2e/best.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    demo = SignLanguageDemo(args.checkpoint, args.device)
    demo.run(args.camera)


if __name__ == '__main__':
    main()

