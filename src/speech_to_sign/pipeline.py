# -*- coding: utf-8 -*-
"""
Speech-to-Sign Full Pipeline

End-to-end pipeline for converting speech/text to sign language video.
"""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path

from .text_to_gloss import TextToGlossModel, RuleBasedTextToGloss, TextTokenizer
from .gloss_retriever import GlossVideoRetriever


class SpeechToSignPipeline:
    """
    End-to-end Speech/Text to Sign Language pipeline.
    
    Pipeline:
    1. [Optional] Speech → Text (using Whisper or other ASR)
    2. Text → Gloss (neural or rule-based translation)
    3. Gloss → Video (retrieval-based synthesis)
    
    Usage:
        pipeline = SpeechToSignPipeline.from_pretrained(checkpoint_dir)
        result = pipeline("Morgen gibt es Regen im Norden")
        video = result['video']  # numpy array (T, H, W, C)
    """
    
    def __init__(
        self,
        text_to_gloss: Union[TextToGlossModel, RuleBasedTextToGloss],
        gloss_retriever: GlossVideoRetriever,
        text_tokenizer: Optional[TextTokenizer] = None,
        gloss_vocab: Dict[str, int] = None,
        asr_model: Optional[any] = None,
        device: str = 'cpu'
    ):
        """
        Args:
            text_to_gloss: Text-to-gloss model (neural or rule-based)
            gloss_retriever: Video retrieval system
            text_tokenizer: Tokenizer for text input (required for neural model)
            gloss_vocab: Gloss vocabulary
            asr_model: Optional ASR model for speech input
            device: Device for neural models
        """
        self.text_to_gloss = text_to_gloss
        self.gloss_retriever = gloss_retriever
        self.text_tokenizer = text_tokenizer
        self.gloss_vocab = gloss_vocab or {}
        self.idx2gloss = {v: k for k, v in self.gloss_vocab.items()}
        self.asr_model = asr_model
        self.device = device
        
        # Move neural model to device if applicable
        if isinstance(self.text_to_gloss, TextToGlossModel):
            self.text_to_gloss = self.text_to_gloss.to(device)
            self.text_to_gloss.eval()
    
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str,
        data_dir: str,
        use_neural: bool = True,
        device: str = 'cpu'
    ) -> 'SpeechToSignPipeline':
        """
        Load pipeline from pretrained checkpoints.
        
        Args:
            checkpoint_dir: Directory containing model checkpoints
            data_dir: Path to PHOENIX data directory
            use_neural: Whether to use neural T2G model (False = rule-based)
            device: Device for inference
            
        Returns:
            Configured pipeline
        """
        import json
        
        checkpoint_dir = Path(checkpoint_dir)
        data_dir = Path(data_dir)
        
        # Load gloss vocabulary
        gloss_vocab_path = checkpoint_dir / 'vocab.json'
        if gloss_vocab_path.exists():
            with open(gloss_vocab_path, 'r', encoding='utf-8') as f:
                gloss_vocab = json.load(f)
        else:
            gloss_vocab = {}
        
        # Load text-to-gloss model
        if use_neural:
            t2g_checkpoint = checkpoint_dir / 'text2gloss.pth'
            t2g_config = checkpoint_dir / 'text2gloss_config.json'
            
            if t2g_checkpoint.exists() and t2g_config.exists():
                with open(t2g_config, 'r') as f:
                    config = json.load(f)
                
                text_to_gloss = TextToGlossModel(**config)
                text_to_gloss.load_state_dict(torch.load(t2g_checkpoint, map_location=device))
                
                text_tokenizer = TextTokenizer.load(checkpoint_dir / 'text_vocab.json')
            else:
                print("Neural T2G model not found, falling back to rule-based")
                text_to_gloss = RuleBasedTextToGloss(gloss_vocab)
                text_tokenizer = None
        else:
            text_to_gloss = RuleBasedTextToGloss(gloss_vocab)
            text_tokenizer = None
        
        # Load gloss retriever
        index_path = data_dir / 'gloss_video_index.pkl'
        gloss_retriever = GlossVideoRetriever(
            str(data_dir / 'phoenix2014-release'),
            index_path=str(index_path) if index_path.exists() else None
        )
        
        return cls(
            text_to_gloss=text_to_gloss,
            gloss_retriever=gloss_retriever,
            text_tokenizer=text_tokenizer,
            gloss_vocab=gloss_vocab,
            device=device
        )
    
    @classmethod
    def from_rule_based(
        cls,
        data_dir: str,
        gloss_vocab: Dict[str, int] = None
    ) -> 'SpeechToSignPipeline':
        """
        Create pipeline with rule-based T2G (no training required).
        
        Args:
            data_dir: Path to PHOENIX data directory
            gloss_vocab: Optional gloss vocabulary
            
        Returns:
            Pipeline with rule-based translation
        """
        import json
        
        data_dir = Path(data_dir)
        
        # Try to load vocab
        if gloss_vocab is None:
            vocab_paths = [
                data_dir / 'checkpoints' / 'hybrid' / 'vocab.json',
                data_dir / 'checkpoints' / 'transformer' / 'vocab.json'
            ]
            for path in vocab_paths:
                if path.exists():
                    with open(path, 'r', encoding='utf-8') as f:
                        gloss_vocab = json.load(f)
                    break
        
        # Create rule-based T2G
        text_to_gloss = RuleBasedTextToGloss(gloss_vocab)
        
        # Create retriever
        phoenix_dir = data_dir / 'data' / 'phoenix2014-release'
        if not phoenix_dir.exists():
            phoenix_dir = data_dir / 'phoenix2014-release'
        
        index_path = data_dir / 'gloss_video_index.pkl'
        
        gloss_retriever = GlossVideoRetriever(
            str(phoenix_dir),
            index_path=str(index_path) if index_path.exists() else None
        )
        
        return cls(
            text_to_gloss=text_to_gloss,
            gloss_retriever=gloss_retriever,
            gloss_vocab=gloss_vocab or {},
            device='cpu'
        )
    
    def transcribe_speech(self, audio: np.ndarray) -> str:
        """
        Transcribe speech to text using ASR.
        
        Args:
            audio: Audio waveform as numpy array
            
        Returns:
            Transcribed text
        """
        if self.asr_model is None:
            try:
                import whisper
                self.asr_model = whisper.load_model("base")
            except ImportError:
                raise ImportError(
                    "Whisper not installed. Install with: pip install openai-whisper"
                )
        
        # Whisper expects audio at 16kHz
        result = self.asr_model.transcribe(
            audio,
            language='de',
            task='transcribe'
        )
        
        return result['text']
    
    def text_to_glosses(self, text: str) -> List[str]:
        """
        Convert text to gloss sequence.
        
        Args:
            text: Input German text
            
        Returns:
            List of gloss tokens
        """
        if isinstance(self.text_to_gloss, RuleBasedTextToGloss):
            return self.text_to_gloss.translate(text)
        
        # Neural model
        if self.text_tokenizer is None:
            raise ValueError("Text tokenizer required for neural model")
        
        # Tokenize input
        text_ids = self.text_tokenizer.encode(text)
        text_tensor = torch.tensor([text_ids], dtype=torch.long, device=self.device)
        
        # Generate gloss IDs
        with torch.no_grad():
            gloss_ids = self.text_to_gloss.generate(text_tensor, max_len=50)
        
        # Decode to gloss strings
        gloss_ids = gloss_ids[0].cpu().tolist()
        glosses = []
        
        for gid in gloss_ids:
            if gid in self.idx2gloss:
                gloss = self.idx2gloss[gid]
                if gloss not in ['<pad>', '<sos>', '<eos>', '<unk>', '<blank>']:
                    glosses.append(gloss)
        
        return glosses
    
    def glosses_to_video(
        self,
        glosses: List[str],
        selection: str = 'random'
    ) -> np.ndarray:
        """
        Convert gloss sequence to video.
        
        Args:
            glosses: List of gloss tokens
            selection: Clip selection method ('random', 'first', 'longest')
            
        Returns:
            Video frames as numpy array (T, H, W, C)
        """
        return self.gloss_retriever.retrieve(glosses, selection=selection)
    
    def __call__(
        self,
        input_data: Union[str, np.ndarray],
        output_path: str = None,
        return_intermediates: bool = True
    ) -> Dict:
        """
        Run full pipeline.
        
        Args:
            input_data: Text string or audio waveform
            output_path: Optional path to save video
            return_intermediates: Whether to return intermediate results
            
        Returns:
            Dictionary with results:
                - 'text': Input/transcribed text
                - 'glosses': Gloss sequence
                - 'video': Video frames (T, H, W, C)
                - 'video_path': Path to saved video (if output_path provided)
        """
        # Step 1: Handle input
        if isinstance(input_data, np.ndarray):
            # Audio input - transcribe
            text = self.transcribe_speech(input_data)
        else:
            text = input_data
        
        # Step 2: Text to Gloss
        glosses = self.text_to_glosses(text)
        
        # Step 3: Gloss to Video
        video = self.glosses_to_video(glosses)
        
        result = {
            'text': text,
            'glosses': glosses,
            'video': video
        }
        
        # Save video if path provided
        if output_path:
            video_path = self.gloss_retriever.retrieve_as_video(
                glosses,
                output_path
            )
            result['video_path'] = video_path
        
        return result
    
    def save(self, checkpoint_dir: str):
        """Save pipeline components."""
        import json
        
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save gloss vocab
        with open(checkpoint_dir / 'vocab.json', 'w', encoding='utf-8') as f:
            json.dump(self.gloss_vocab, f, ensure_ascii=False, indent=2)
        
        # Save neural model if applicable
        if isinstance(self.text_to_gloss, TextToGlossModel):
            torch.save(
                self.text_to_gloss.state_dict(),
                checkpoint_dir / 'text2gloss.pth'
            )
        
        if self.text_tokenizer:
            self.text_tokenizer.save(checkpoint_dir / 'text_vocab.json')
        
        print(f"Pipeline saved to {checkpoint_dir}")


def quick_demo(data_dir: str, text: str = "Morgen gibt es Regen im Norden"):
    """
    Quick demonstration of the pipeline.
    
    Args:
        data_dir: Path to PHOENIX-SLR project directory
        text: German text to convert
    """
    print(f"\n{'='*60}")
    print("Speech-to-Sign Quick Demo")
    print(f"{'='*60}\n")
    
    # Create pipeline with rule-based T2G
    print("Loading pipeline...")
    pipeline = SpeechToSignPipeline.from_rule_based(data_dir)
    
    # Check if index exists
    if len(pipeline.gloss_retriever.gloss_index) == 0:
        print("\nBuilding video index (first run only)...")
        from .gloss_retriever import build_default_index
        phoenix_dir = Path(data_dir) / 'data' / 'phoenix2014-release'
        if phoenix_dir.exists():
            index_path = Path(data_dir) / 'gloss_video_index.pkl'
            pipeline.gloss_retriever = build_default_index(
                str(phoenix_dir),
                str(index_path)
            )
    
    # Run pipeline
    print(f"\nInput text: {text}")
    result = pipeline(text)
    
    print(f"Glosses: {' → '.join(result['glosses'])}")
    print(f"Video shape: {result['video'].shape}")
    print(f"Video duration: {result['video'].shape[0] / 25:.1f}s @ 25fps")
    
    # Save sample video
    output_path = Path(data_dir) / 'sample_output.mp4'
    pipeline.gloss_retriever.retrieve_as_video(
        result['glosses'],
        str(output_path)
    )
    print(f"\nVideo saved to: {output_path}")
    
    return result

