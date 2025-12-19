# -*- coding: utf-8 -*-
"""Quick test script for Speech-to-Sign pipeline."""

from src.speech_to_sign.pipeline import SpeechToSignPipeline
import cv2
import numpy as np

def main():
    print("="*60)
    print("Speech-to-Sign Pipeline Test")
    print("="*60)
    
    # Load the pipeline
    print("\nLoading pipeline...")
    pipeline = SpeechToSignPipeline.from_rule_based('.')
    
    print(f"Glosses available: {len(pipeline.gloss_retriever.gloss_index)}")
    
    # Test cases
    test_inputs = [
        "Morgen Regen Nord",
        "Heute Sonne warm",
        "Schnee Alpen",
        "Wind stark Kueste",
        "Temperatur zwanzig Grad"
    ]
    
    for text in test_inputs:
        print(f"\n{'='*40}")
        print(f"Input: {text}")
        
        result = pipeline(text)
        
        print(f"Glosses: {' -> '.join(result['glosses'])}")
        print(f"Video shape: {result['video'].shape}")
        print(f"Duration: {result['video'].shape[0] / 25:.1f}s @ 25fps")
    
    # Save a sample video
    print("\n" + "="*60)
    print("Saving sample video...")
    
    result = pipeline("Morgen Regen Schnee", output_path="sample_sign_output.mp4")
    print(f"Video saved to: sample_sign_output.mp4")
    
    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)

if __name__ == "__main__":
    main()


