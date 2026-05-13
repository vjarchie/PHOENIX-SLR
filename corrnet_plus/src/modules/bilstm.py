"""
BiLSTM Layer for CorrNet+ temporal modeling.
Based on: https://github.com/hulianyuyy/CorrNet_Plus
"""
import torch
import torch.nn as nn


class BiLSTMLayer(nn.Module):
    """Bidirectional LSTM layer for sequence modeling."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        rnn_type: str = 'LSTM'
    ):
        super().__init__()
        
        self.dropout = dropout
        self.num_layers = num_layers
        self.input_size = input_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size // self.num_directions
        self.rnn_type = rnn_type
        
        self.rnn = getattr(nn, self.rnn_type)(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        )
    
    def forward(
        self,
        src_feats: torch.Tensor,
        src_lens: torch.Tensor,
        hidden: torch.Tensor = None
    ) -> dict:
        """
        Args:
            src_feats: (max_src_len, batch_size, D)
            src_lens: (batch_size,)
            hidden: optional initial hidden state
            
        Returns:
            dict with:
                - predictions: (max_src_len, batch_size, hidden_size * num_directions)
                - hidden: (num_layers, batch_size, hidden_size * num_directions)
        """
        packed_emb = nn.utils.rnn.pack_padded_sequence(
            src_feats, src_lens.cpu(), enforce_sorted=False
        )
        
        if hidden is not None and self.rnn_type == 'LSTM':
            half = hidden.size(0) // 2
            hidden = (hidden[:half], hidden[half:])
        
        packed_outputs, hidden = self.rnn(packed_emb, hidden)
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        
        if self.bidirectional:
            hidden = self._cat_directions(hidden)
        
        if isinstance(hidden, tuple):
            hidden = torch.cat(hidden, 0)
        
        return {
            "predictions": rnn_outputs,
            "hidden": hidden
        }
    
    def _cat_directions(self, hidden):
        """Concatenate forward and backward hidden states."""
        def _cat(h):
            return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        
        if isinstance(hidden, tuple):
            hidden = tuple([_cat(h) for h in hidden])
        else:
            hidden = _cat(hidden)
        
        return hidden
