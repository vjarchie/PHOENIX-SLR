# -*- coding: utf-8 -*-
"""Test video output generation."""

from src.speech_to_sign.pipeline import SpeechToSignPipeline
from PIL import Image
import numpy as np

pipeline = SpeechToSignPipeline.from_rule_based('.')
result = pipeline('Morgen Regen Nord')

print(f"Generated {result['video'].shape[0]} frames")
print(f"Frame shape: {result['video'].shape[1:]}")

# Save first frame as image to verify
frame = result['video'][0]
img = Image.fromarray(frame)
img.save('test_frame.png')
print("Saved test_frame.png")

# Create a quick GIF
frames_pil = [Image.fromarray(f) for f in result['video'][::2]]
frames_pil[0].save('test_sign.gif', save_all=True, append_images=frames_pil[1:], duration=80, loop=0)
print("Saved test_sign.gif")

print("\nOpen test_frame.png and test_sign.gif to see the output!")


