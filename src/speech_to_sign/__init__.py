# -*- coding: utf-8 -*-
"""
Speech-to-Sign Language Production (SLP) Module

Converts text/speech input to sign language video output.
Reverses the SLR (Sign Language Recognition) pipeline.

Components:
- TextToGloss: Translates German text to DGS gloss sequences
- GlossVideoRetriever: Retrieves and concatenates video clips
- SignVideoGenerator: Generates video frames from glosses (optional)
- SpeechToSignPipeline: End-to-end pipeline
"""

from .text_to_gloss import TextToGlossModel, RuleBasedTextToGloss
from .gloss_retriever import GlossVideoRetriever
from .pipeline import SpeechToSignPipeline

__all__ = [
    'TextToGlossModel',
    'RuleBasedTextToGloss', 
    'GlossVideoRetriever',
    'SpeechToSignPipeline'
]

