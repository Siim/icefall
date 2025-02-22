import k2
import torch
import torch.nn as nn
from typing import Optional, List, Dict, Tuple, Union
from pathlib import Path
import logging
import copy

logger = logging.getLogger(__name__)

def create_estonian_token_table(vocab_file: str) -> k2.SymbolTable:
    """Create a token table for Estonian vocabulary"""
    vocab_path = Path(vocab_file)
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_file}")
    
    # Create symbol table - k2 automatically adds <eps> with ID 0
    token_table = k2.SymbolTable()
    
    # Add special tokens starting from ID 1
    token_table.add("<blk>", 1)  # blank token
    token_table.add("<sos/eos>", 2)  # start/end of sequence
    
    # Add Estonian characters
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for line in f:
            token = line.strip()
            if token and token not in ["<blk>", "<sos/eos>"]:
                token_table.add(token)
    
    logger.info(f"Created Estonian token table with {len(token_table)} tokens")
    return token_table

def create_estonian_decoding_graph(
    token_table: k2.SymbolTable,
    num_tokens: int,
    device: torch.device
) -> k2.Fsa:
    """Create a decoding graph for Estonian ASR"""
    # Create a trivial graph for CTC-like decoding
    graph = k2.trivial_graph(num_tokens, device=device)
    
    # Add self-loops for blank transitions
    graph = k2.add_epsilon_self_loops(graph)
    
    # Remove unnecessary states and arcs
    graph = k2.connect(graph)
    graph = k2.arc_sort(graph)
    
    return graph

class EstonianDecoder(nn.Module):
    """Decoder for Estonian ASR with FSA-based decoding"""
    
    def __init__(
        self,
        vocab_size: int,
        decoder_dim: int,
        blank_id: int = 0,
        context_size: int = 2,  # From paper: using bigram context
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.decoder_dim = decoder_dim
        self.blank_id = blank_id
        self.context_size = context_size
        
        # Embedding layer for tokens
        self.embedding = nn.Embedding(vocab_size, decoder_dim)
        
        # Initialize with Xavier (as per paper)
        nn.init.xavier_uniform_(self.embedding.weight)
        
        # Decoder layers (following paper's architecture)
        self.layers = nn.ModuleList([
            nn.Linear(decoder_dim, decoder_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Paper uses dropout for regularization
            nn.Linear(decoder_dim, decoder_dim)
        ])
        
        # Initialize FSA objects as None
        self._decoding_graph = None
        self._token_table = None
    
    def __deepcopy__(self, memo):
        """Custom deepcopy implementation to handle k2 FSA objects"""
        # Create a new instance without FSA objects
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        
        # Copy all attributes except FSA objects
        for k, v in self.__dict__.items():
            if k not in ['_decoding_graph', '_token_table']:
                setattr(result, k, torch.nn.Parameter(v.clone()) if isinstance(v, torch.nn.Parameter) else copy.deepcopy(v, memo))
        
        # Set FSA objects to None in the copy
        result._decoding_graph = None
        result._token_table = None
        
        return result
    
    def forward(
        self,
        y: Union[torch.Tensor, k2.RaggedTensor],
        need_pad: bool = False
    ) -> torch.Tensor:
        """
        Args:
            y: A 2-D tensor of shape (N, U) with U <= context_size
               or a RaggedTensor containing token sequences
            need_pad: If True, pad y with blank_id to ensure context_size
        Returns:
            A 2-D tensor of shape (N, decoder_dim)
        """
        if isinstance(y, k2.RaggedTensor):
            # Convert RaggedTensor to dense tensor
            dense_list = []
            for i in range(y.dim0()):
                # Get values for this sequence
                seq = y[i].values.tolist()
                dense_list.append(seq)
            
            # Pad sequences to same length
            max_len = max(len(seq) for seq in dense_list)
            y_padded = []
            for seq in dense_list:
                if len(seq) < max_len:
                    seq = seq + [self.blank_id] * (max_len - len(seq))
                y_padded.append(seq)
            
            # Convert to tensor on same device as RaggedTensor
            y = torch.tensor(y_padded, device=y.device)
        
        # Handle padding if needed (for context size)
        if need_pad and y.shape[1] < self.context_size:
            padding = torch.full(
                (y.shape[0], self.context_size - y.shape[1]),
                self.blank_id,
                device=y.device,
                dtype=y.dtype
            )
            y = torch.cat([padding, y], dim=1)
        
        # Embed tokens
        embedded = self.embedding(y)
        
        # Average embeddings (as per paper's decoder architecture)
        decoder_out = torch.mean(embedded, dim=1)
        
        # Apply decoder layers
        for layer in self.layers:
            decoder_out = layer(decoder_out)
        
        return decoder_out

    @property
    def decoding_graph(self):
        """Getter for decoding graph"""
        return self._decoding_graph
    
    @decoding_graph.setter
    def decoding_graph(self, graph):
        """Setter for decoding graph"""
        self._decoding_graph = graph
    
    @property
    def token_table(self):
        """Getter for token table"""
        return self._token_table
    
    @token_table.setter
    def token_table(self, table):
        """Setter for token table"""
        self._token_table = table

def fast_beam_search_one_best(
    model: nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    beam: int,
    max_states: int,
    max_contexts: int,
    decoding_graph: Optional[k2.Fsa] = None,
) -> List[List[int]]:
    """
    Fast beam search to find the best path through the decoding graph.
    
    Args:
        model: The transducer model
        encoder_out: Output from the encoder (N, T, C)
        encoder_out_lens: Length of each sequence
        beam: Beam size for search
        max_states: Max FSA states to keep
        max_contexts: Max right contexts to keep
        decoding_graph: Optional FSA decoding graph
    Returns:
        List of token IDs for each sequence
    """
    device = encoder_out.device
    batch_size = encoder_out.size(0)
    
    if decoding_graph is None:
        decoding_graph = k2.trivial_graph(model.decoder.vocab_size, device=device)
    
    # Initialize hypothesis
    hyps = []
    
    # Process each sequence in batch
    for i in range(batch_size):
        # Get encoder output for this sequence
        enc_out = encoder_out[i, :encoder_out_lens[i]].unsqueeze(0)
        
        # Create lattice
        lattice = k2.intersect_dense(
            decoding_graph,
            enc_out,
            output_beam=beam,
            max_states=max_states,
            max_contexts=max_contexts
        )
        
        # Find best path
        best_path = k2.shortest_path(lattice, use_double_scores=True)
        
        # Convert path to token IDs
        token_ids = []
        for arc in best_path.arcs:
            if arc.label != 0:  # Skip blank tokens
                token_ids.append(arc.label)
        
        hyps.append(token_ids)
    
    return hyps 