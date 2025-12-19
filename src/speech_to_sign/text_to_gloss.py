# -*- coding: utf-8 -*-
"""
Text-to-Gloss Translation Module

Converts German text to DGS (German Sign Language) gloss sequences.
Uses the same vocabulary as the SLR model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TextToGlossModel(nn.Module):
    """
    Neural Text-to-Gloss translation model.
    
    Architecture: Transformer Encoder-Decoder
    - Encoder: Processes German text tokens
    - Decoder: Generates gloss sequence autoregressively
    """
    
    def __init__(
        self,
        text_vocab_size: int,
        gloss_vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 100,
        pad_idx: int = 0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        # Text encoder
        self.text_embedding = nn.Embedding(text_vocab_size, d_model, padding_idx=pad_idx)
        self.text_pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Gloss decoder
        self.gloss_embedding = nn.Embedding(gloss_vocab_size, d_model, padding_idx=pad_idx)
        self.gloss_pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, gloss_vocab_size)
        
        self._init_weights()
        
        print(f"TextToGlossModel initialized:")
        print(f"  Text vocab: {text_vocab_size}")
        print(f"  Gloss vocab: {gloss_vocab_size}")
        print(f"  d_model: {d_model}")
        print(f"  Params: {sum(p.numel() for p in self.parameters()):,}")
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)
        return mask
    
    def encode(self, text_ids: torch.Tensor, text_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """Encode text to features."""
        x = self.text_embedding(text_ids) * math.sqrt(self.d_model)
        x = self.text_pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=text_padding_mask)
        return x
    
    def decode(
        self,
        gloss_ids: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        memory_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Decode gloss sequence from encoder memory."""
        x = self.gloss_embedding(gloss_ids) * math.sqrt(self.d_model)
        x = self.gloss_pos_encoder(x)
        
        if tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(
                gloss_ids.size(1), gloss_ids.device
            )
        
        x = self.decoder(
            x, memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_padding_mask
        )
        
        logits = self.output_proj(x)
        return logits
    
    def forward(
        self,
        text_ids: torch.Tensor,
        gloss_ids: torch.Tensor,
        text_padding_mask: torch.Tensor = None,
        gloss_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass for training.
        
        Args:
            text_ids: (batch, text_len) - input text token IDs
            gloss_ids: (batch, gloss_len) - target gloss IDs (shifted right)
            text_padding_mask: (batch, text_len) - True for padding
            gloss_padding_mask: (batch, gloss_len) - True for padding
            
        Returns:
            logits: (batch, gloss_len, gloss_vocab_size)
        """
        memory = self.encode(text_ids, text_padding_mask)
        logits = self.decode(gloss_ids, memory, memory_padding_mask=text_padding_mask)
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        text_ids: torch.Tensor,
        max_len: int = 50,
        sos_token: int = 3,
        eos_token: int = 4,
        text_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Generate gloss sequence from text (greedy decoding).
        
        Args:
            text_ids: (batch, text_len) - input text
            max_len: maximum output length
            sos_token: start of sequence token
            eos_token: end of sequence token
            
        Returns:
            generated: (batch, seq_len) - generated gloss IDs
        """
        self.eval()
        batch_size = text_ids.size(0)
        device = text_ids.device
        
        # Encode text
        memory = self.encode(text_ids, text_padding_mask)
        
        # Start with <sos>
        generated = torch.full((batch_size, 1), sos_token, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_len - 1):
            logits = self.decode(generated, memory, memory_padding_mask=text_padding_mask)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            finished = finished | (next_token.squeeze(-1) == eos_token)
            if finished.all():
                break
        
        return generated


class RuleBasedTextToGloss:
    """
    Simple rule-based text-to-gloss conversion for German weather domain.
    
    Uses keyword matching and word order rules specific to DGS.
    This is a fallback for when neural model isn't trained.
    """
    
    # German word to DGS gloss mapping (weather domain)
    WORD_TO_GLOSS = {
        # Greetings
        'hallo': 'HALLO',
        'guten': 'GUT',
        'abend': 'ABEND',
        'morgen': 'MORGEN',
        'tag': 'TAG',
        'nacht': 'NACHT',
        'tschüss': 'TSCHUESS',
        
        # Weather
        'wetter': 'WETTER',
        'regen': 'REGEN',
        'schnee': 'SCHNEE',
        'sonne': 'SONNE',
        'sonnig': 'SONNE',
        'wolke': 'WOLKE',
        'wolken': 'WOLKE',
        'bewölkt': 'BEWOELKT',
        'nebel': 'NEBEL',
        'gewitter': 'GEWITTER',
        'sturm': 'STURM',
        'wind': 'WIND',
        'windig': 'WIND',
        'frost': 'FROST',
        'warm': 'WARM',
        'kalt': 'KALT',
        'kühl': 'KUEHL',
        'heiß': 'HEISS',
        'trocken': 'TROCKEN',
        'feucht': 'FEUCHT',
        'schauer': 'SCHAUER',
        
        # Directions
        'nord': 'NORD',
        'norden': 'NORD',
        'süd': 'SUED',
        'süden': 'SUED',
        'ost': 'OST',
        'osten': 'OST',
        'west': 'WEST',
        'westen': 'WEST',
        'nordost': 'NORDOST',
        'nordwest': 'NORDWEST',
        'südost': 'SUEDOST',
        'südwest': 'SUEDWEST',
        
        # Time
        'heute': 'HEUTE',
        'gestern': 'GESTERN',
        'übermorgen': 'UEBERMORGEN',
        'woche': 'WOCHE',
        'wochenende': 'WOCHENENDE',
        'montag': 'MONTAG',
        'dienstag': 'DIENSTAG',
        'mittwoch': 'MITTWOCH',
        'donnerstag': 'DONNERSTAG',
        'freitag': 'FREITAG',
        'samstag': 'SAMSTAG',
        'sonntag': 'SONNTAG',
        
        # Locations
        'deutschland': 'DEUTSCHLAND',
        'bayern': 'BAYERN',
        'berlin': 'BERLIN',
        'hamburg': 'HAMBURG',
        'münchen': 'MUENCHEN',
        'alpen': 'ALPEN',
        'küste': 'KUESTE',
        'meer': 'MEER',
        'see': 'SEE',
        'berg': 'BERG',
        
        # Numbers (as words)
        'null': 'NULL',
        'eins': 'EINS',
        'zwei': 'ZWEI',
        'drei': 'DREI',
        'vier': 'VIER',
        'fünf': 'FUENF',
        'sechs': 'SECHS',
        'sieben': 'SIEBEN',
        'acht': 'ACHT',
        'neun': 'NEUN',
        'zehn': 'ZEHN',
        'zwanzig': 'ZWANZIG',
        'dreißig': 'DREISSIG',
        
        # Quantities/Intensifiers
        'viel': 'VIEL',
        'wenig': 'WENIG',
        'stark': 'STARK',
        'schwach': 'SCHWACH',
        'leicht': 'LEICHT',
        'mehr': 'MEHR',
        'weniger': 'WENIGER',
        'bisschen': 'BISSCHEN',
        
        # Connectors
        'und': 'UND',
        'oder': 'ODER',
        'aber': 'ABER',
        'dann': 'DANN',
        'auch': 'AUCH',
        'noch': 'NOCH',
        'schon': 'SCHON',
        
        # Other
        'grad': 'GRAD',
        'temperatur': 'TEMPERATUR',
        'maximal': 'MAXIMAL',
        'minimal': 'MINUS',
        'bis': 'BIS',
        'etwa': 'UNGEFAEHR',
        'ungefähr': 'UNGEFAEHR',
    }
    
    def __init__(self, gloss_vocab: Dict[str, int] = None):
        """
        Args:
            gloss_vocab: Gloss vocabulary mapping (for validation)
        """
        self.gloss_vocab = gloss_vocab or {}
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for number extraction."""
        self.number_pattern = re.compile(r'(\d+)')
        self.temperature_pattern = re.compile(r'(-?\d+)\s*(?:grad|°)', re.IGNORECASE)
    
    def _normalize_text(self, text: str) -> str:
        """Normalize German text for matching."""
        text = text.lower()
        # German-specific replacements
        text = text.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue')
        text = text.replace('ß', 'ss')
        return text
    
    def _extract_numbers(self, text: str) -> List[str]:
        """Extract numbers and convert to gloss format."""
        glosses = []
        
        # Temperature patterns
        temps = self.temperature_pattern.findall(text)
        for temp in temps:
            if int(temp) < 0:
                glosses.append('MINUS')
            glosses.append(f"{abs(int(temp))}")  # Will be converted to word
            glosses.append('GRAD')
        
        return glosses
    
    def translate(self, text: str) -> List[str]:
        """
        Translate German text to DGS gloss sequence.
        
        Args:
            text: German input text
            
        Returns:
            List of gloss tokens
        """
        # Normalize
        text_lower = text.lower()
        text_norm = self._normalize_text(text)
        
        # Extract words
        words = re.findall(r'\b\w+\b', text_norm)
        
        glosses = []
        
        for word in words:
            # Check direct mapping
            if word in self.WORD_TO_GLOSS:
                gloss = self.WORD_TO_GLOSS[word]
                # Validate against vocabulary if available
                if not self.gloss_vocab or gloss in self.gloss_vocab:
                    glosses.append(gloss)
            # Check if it's a number
            elif word.isdigit():
                num = int(word)
                if num <= 20:
                    # Convert to word gloss
                    num_words = {
                        0: 'NULL', 1: 'EINS', 2: 'ZWEI', 3: 'DREI', 4: 'VIER',
                        5: 'FUENF', 6: 'SECHS', 7: 'SIEBEN', 8: 'ACHT', 9: 'NEUN',
                        10: 'ZEHN', 11: 'ELF', 12: 'ZWOELF', 13: 'DREIZEHN',
                        14: 'VIERZEHN', 15: 'FUENFZEHN', 16: 'SECHSZEHN',
                        17: 'SIEBZEHN', 18: 'ACHTZEHN', 19: 'NEUNZEHN', 20: 'ZWANZIG'
                    }
                    if num in num_words:
                        glosses.append(num_words[num])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_glosses = []
        for g in glosses:
            if g not in seen:
                seen.add(g)
                unique_glosses.append(g)
        
        return unique_glosses if unique_glosses else ['UNK']
    
    def __call__(self, text: str) -> List[str]:
        return self.translate(text)


class TextTokenizer:
    """
    Simple text tokenizer for German text.
    Uses word-level tokenization with special tokens.
    """
    
    def __init__(self, vocab: Dict[str, int] = None):
        self.vocab = vocab or {
            '<pad>': 0,
            '<unk>': 1,
            '<sos>': 2,
            '<eos>': 3
        }
        self.idx2word = {v: k for k, v in self.vocab.items()}
    
    def build_vocab(self, texts: List[str], min_freq: int = 1):
        """Build vocabulary from list of texts."""
        word_counts = {}
        
        for text in texts:
            words = self._tokenize(text)
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        for word, count in sorted(word_counts.items(), key=lambda x: -x[1]):
            if count >= min_freq and word not in self.vocab:
                self.vocab[word] = len(self.vocab)
        
        self.idx2word = {v: k for k, v in self.vocab.items()}
        print(f"Built vocabulary with {len(self.vocab)} tokens")
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text to words."""
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        return words
    
    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """Encode text to token IDs."""
        words = self._tokenize(text)
        ids = [self.vocab.get(w, self.vocab['<unk>']) for w in words]
        
        if add_special:
            ids = [self.vocab['<sos>']] + ids + [self.vocab['<eos>']]
        
        return ids
    
    def decode(self, ids: List[int], remove_special: bool = True) -> str:
        """Decode token IDs to text."""
        words = [self.idx2word.get(i, '<unk>') for i in ids]
        
        if remove_special:
            words = [w for w in words if w not in ['<pad>', '<sos>', '<eos>']]
        
        return ' '.join(words)
    
    def save(self, path: str):
        """Save vocabulary to file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TextTokenizer':
        """Load vocabulary from file."""
        with open(path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        return cls(vocab)

