import torch,os
from torch import nn
import math
from torch.nn.init import xavier_uniform_
import copy
from torch import Tensor
from typing import Optional

from torch.nn import functional as F

class DecoderResBlock(nn.Module):
    """
    A simple residual block with 1x1, 3x3, then 1x1 convolutions.
    Includes BatchNorm and ReLU activations.
    """
    def __init__(self, inchannel: int, outchannel: int, stride: int = 1, shortcut=None):
        super(DecoderResBlock, self).__init__()
        self.left = nn.Sequential(
                nn.Conv2d(inchannel,int(outchannel/2),kernel_size = 1),
                nn.BatchNorm2d(int(outchannel/2)),
                nn.ReLU(),
                nn.Conv2d(int(outchannel/2), int(outchannel / 2), kernel_size = 3, stride=stride, padding=1),
                nn.BatchNorm2d(int(outchannel / 2)),
                nn.ReLU(),
                nn.Conv2d(int(outchannel/2),outchannel,kernel_size = 1),
                nn.BatchNorm2d(outchannel)
        )
        # `shortcut` argument is not used in the current implementation of forward.
        # If a custom shortcut connection is needed, it should be incorporated into the forward pass.
        self.right = shortcut

    def forward(self, x: Tensor) -> Tensor:
        out = self.left(x)
        residual = x # Standard residual connection
        out = out + residual
        return F.relu(out)
    
class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the tokens in the sequence.
    The positional encodings have the same dimension as the embeddings so that the two can be summed.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: [max_len, 1, d_model]
        self.register_buffer('pe', pe) # Not a model parameter, but should be part of state_dict

        # Optional: Learnable 1D positional embedding.
        # This is initialized but not used in the current forward pass.
        # It could be an alternative or an addition to the fixed sinusoidal encoding.
        # The size 52 might be a specific max_len for this learnable embedding.
        self.embedding_1D = nn.Embedding(52, int(d_model))

    def forward(self, x: Tensor) -> Tensor:
        # Add fixed sinusoidal positional encoding up to the length of x.
        # x shape: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]

        # Example of using the learnable positional embedding (currently commented out):
        # `arange_len = x.size(0)` (if 52 is max length of this embedding)
        # `learnable_pe = self.embedding_1D(torch.arange(arange_len).to(x.device))`
        # `learnable_pe = learnable_pe.unsqueeze(1).repeat(1, x.size(1), 1)` # Adjust shape
        # `x = x + learnable_pe`

        return self.dropout(x)

class Mesh_TransformerDecoderLayer(nn.Module):
    """
    A single layer of the Transformer decoder, adapted for mesh-like data or sequences.
    Includes self-attention, multi-head cross-attention with memory, and a feedforward network.
    """
    __constants__ = ['batch_first', 'norm_first'] # Kept from original, though batch_first is not used in constructor
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False, # batch_first not used
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype} # For potential future use
        super(Mesh_TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout) # d_model was int(d_model)

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout) # General dropout for feedforward
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first # If True, apply LayerNorm before attention/FFN, else after.
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout) # Dropout after self-attention
        self.dropout2 = nn.Dropout(dropout) # Dropout after multihead-attention
        self.dropout3 = nn.Dropout(dropout) # Dropout after feedforward network

        self.activation = nn.ReLU() # Activation for feedforward network

        # Note: fc_alpha1, fc_alpha2, fc_alpha3 are initialized but not used in the current forward pass.
        # These might be intended for a gating mechanism or feature fusion that is not implemented here.
        self.fc_alpha1 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha2 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha3 = nn.Linear(d_model + d_model, d_model)

        self.init_weights()

    def init_weights(self):
        # Initialize weights for the (currently unused) fc_alpha layers.
        nn.init.xavier_uniform_(self.fc_alpha1.weight)
        nn.init.xavier_uniform_(self.fc_alpha2.weight)
        nn.init.xavier_uniform_(self.fc_alpha3.weight)
        nn.init.constant_(self.fc_alpha1.bias, 0)
        nn.init.constant_(self.fc_alpha2.bias, 0)
        nn.init.constant_(self.fc_alpha3.bias, 0)


    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        self_att_tgt = self.norm1(tgt + self._sa_block(tgt, tgt_mask, tgt_key_padding_mask))
        # # cross self-attention
        mha_out, _att_weight = self._mha_block(self_att_tgt, memory, memory_mask, memory_key_padding_mask)
        tgt_after_mha = self.norm2(self_att_tgt + mha_out) # Add & Norm for cross-attention

        # Feedforward block
        ff_out = self._ff_block(tgt_after_mha)
        x = self.norm3(tgt_after_mha + ff_out) # Add & Norm for feedforward

        # The original return `x + tgt` is unusual. Standard Transformer decoders output `x`.
        # Adding `tgt` (the initial input to the layer before self-attention) would mean the output
        # strongly carries over the input, which might be intended for very residual behavior,
        # but it deviates from the typical formulation `output = LayerNorm(sublayer_input + Sublayer(sublayer_input))`.
        # If `x` already incorporates `tgt` through residual connections, then just `return x`.
        # Current behavior: output is `LayerNorm(processed_tgt + FF(processed_tgt)) + original_tgt_input_to_layer`.
        # This seems like an extra residual connection applied incorrectly.
        # For standard behavior, it should be `return x`.
        # Keeping `return x + tgt` to match original code, but with this note.
        return x + tgt

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)
 
    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x, att_weight = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=True)
        return self.dropout2(x),  att_weight

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

class StackTransformer(nn.Module):
    r"""StackTransformer is a stack of N decoder layers

    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(StackTransformer, self).__init__()
        self.layers = torch.nn.modules.transformer._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class DecoderTransformerWithMask(nn.Module):
    """
    Decoder with Transformer for processing change prediction mask.
    """
    def __init__(self, encoder_dim, feature_dim, vocab_size, max_lengths, word_vocab, n_head, n_layers, dropout):
        """
        :param n_head: the number of heads in Transformer
        :param n_layers: the number of layers of Transformer
        """
        super(DecoderTransformerWithMask, self).__init__()

        # Parameters
        self.feature_dim = feature_dim
        self.embed_dim = feature_dim
        self.vocab_size = vocab_size
        self.max_lengths = max_lengths
        self.word_vocab = word_vocab
        self.dropout = dropout

        # Embedding layers
        # self.Conv1 = nn.Conv2d(encoder_dim, feature_dim, kernel_size=1) # Original if encoder_dim was variable
        self.Conv1 = nn.Conv2d(encoder_dim, feature_dim, kernel_size=1) # Projects mask to feature_dim
        self.LN = DecoderResBlock(feature_dim, feature_dim) # Residual block for mask features
        self.vocab_embedding = nn.Embedding(vocab_size, self.embed_dim)

        # Transformer layers
        decoder_layer = Mesh_TransformerDecoderLayer(feature_dim, n_head, dim_feedforward=feature_dim * 4, dropout=self.dropout)
        self.transformer = StackTransformer(decoder_layer, n_layers)
        self.position_encoding = PositionalEncoding(feature_dim, max_len=max_lengths)

        # Linear layer to predict the vocabulary over masked change
        self.wdc = nn.Linear(feature_dim, vocab_size)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.cos = torch.nn.CosineSimilarity(dim=1)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """ Initialize some parameters with values from the uniform distribution """
        self.vocab_embedding.weight.data.uniform_(-0.1, 0.1) 
        self.wdc.bias.data.fill_(0)
        self.wdc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, mask: Tensor, encoded_captions: Tensor, caption_lengths: Tensor) -> tuple[Tensor, Tensor, list[int]]:
        """
        Forward pass for training the caption decoder.

        Args:
            mask: The predicted change map from U-Net, shape (batch_size, encoder_dim, H, W).
            encoded_captions: Ground truth captions, tokenized and padded, shape (batch_size, max_caption_length).
            caption_lengths: Actual lengths of captions before padding, shape (batch_size).

        Returns:
            pred: Predicted logits over vocabulary, shape (batch_size, max_caption_length, vocab_size).
            encoded_captions_sorted: Captions sorted by length (for potential PackedSequence use, though not used here).
            decode_lengths_sorted: Sorted caption lengths.
        """
        # Step 1: Process visual features (mask) from U-Net output.
        # This serves as the 'memory' for the Transformer decoder's cross-attention.
        x_visual_features = self.LN(self.Conv1(mask))  # Shape: (batch_size, feature_dim, H, W)
        batch, channel = x_visual_features.size(0), x_visual_features.size(1)
        # Reshape visual features to be sequence-like for Transformer: (H*W, batch_size, feature_dim)
        x_visual_features = x_visual_features.view(batch, channel, -1).permute(2, 0, 1)

        # Step 2: Prepare target caption sequence for Transformer.
        seq_len = encoded_captions.size(1)
        # Create target mask: a triangular mask for auto-regressive decoding (cannot attend to future tokens).
        tgt_autoregressive_mask = torch.triu(torch.ones(seq_len, seq_len, device=mask.device) * float('-inf'), diagonal=1)

        # Create padding mask: masks out <NULL> or <END> padding tokens in the target sequence.
        # This prevents attention mechanism from focusing on padding tokens.
        tgt_padding_mask = (encoded_captions == self.word_vocab['<NULL>']) | \
                           (encoded_captions == self.word_vocab['<END>'])

        # Embed target captions and add positional encoding.
        word_emb = self.vocab_embedding(encoded_captions)  # Shape: (batch_size, seq_len, feature_dim)
        word_emb = word_emb.transpose(0, 1)  # Shape: (seq_len, batch_size, feature_dim) for Transformer
        word_emb = self.position_encoding(word_emb)

        # Step 3: Pass through Transformer decoder.
        # `word_emb` is the target sequence (shifted right during teacher forcing).
        # `x_visual_features` is the memory (encoder output from visual features).
        pred = self.transformer(tgt=word_emb, memory=x_visual_features,
                                tgt_mask=tgt_autoregressive_mask,
                                tgt_key_padding_mask=tgt_padding_mask)  # Shape: (seq_len, batch, feature_dim)

        # Project to vocabulary space.
        pred = self.wdc(self.dropout_layer(pred))  # Shape: (seq_len, batch, vocab_size)
        pred = pred.permute(1, 0, 2) # Shape: (batch, seq_len, vocab_size)

        # Step 4: Sort outputs by caption length.
        # This is often done for PackedSequence used in RNNs, but can also be useful for consistency
        # or if specific loss functions expect sorted inputs (though CrossEntropyLoss handles padding).
        caption_lengths_squeezed = caption_lengths.squeeze(-1) if caption_lengths.ndim > 1 else caption_lengths
        caption_lengths_sorted, sort_ind = caption_lengths_squeezed.sort(dim=0, descending=True)
        encoded_captions_sorted = encoded_captions[sort_ind]
        pred_sorted = pred[sort_ind]

        # Calculate effective lengths for loss (excluding <START> token, as prediction starts from the second token).
        decode_lengths_sorted = (caption_lengths_sorted - 1).tolist()

        return pred_sorted, encoded_captions_sorted, decode_lengths_sorted


    def sample(self, x: Tensor, k: int = 1) -> list[int]:
        """
        Greedy sampling method to generate a caption for the given mask(s).

        Args:
            x: Input tensor, typically the change map from U-Net,
               shape (batch_size, encoder_dim, H, W). Batch size should be 1 for this greedy sample.
            k: Beam size (unused in greedy sampling, kept for compatibility signature with sample1).

        Returns:
            A list of token IDs representing the generated caption for the first sample in the batch.
        """
        # Step 1: Process input mask `x` to get visual features (memory for Transformer).
        x_visual_features = self.LN(self.Conv1(x)) # Shape: (batch_size, feature_dim, H, W)
        batch_size, channel = x_visual_features.size(0), x_visual_features.size(1)
        # Reshape for Transformer: (H*W, batch_size, feature_dim)
        x_visual_features = x_visual_features.view(batch_size, channel, -1).permute(2, 0, 1)

        # Initialize target sequence for decoding. Start with <START> token.
        # Assumes batch_size is 1 for this greedy sampling implementation.
        current_tgt_tokens = torch.zeros(batch_size, self.max_lengths, dtype=torch.long, device=x.device)
        current_tgt_tokens[:, 0] = self.word_vocab['<START>']

        # `generated_sequence_ids` will store the list of generated token IDs for each item in the batch.
        generated_sequence_ids_batch = [[self.word_vocab['<START>']] for _ in range(batch_size)]

        # Auto-regressive decoding loop
        for step in range(self.max_lengths - 1): # Max_lengths-1 steps to generate tokens after <START>
            current_seq_len = step + 1 # Current length of sequences being processed

            # Prepare masks for the Transformer based on the current length of generated sequences.
            # `tgt_autoregressive_mask`: Prevents attention to future tokens.
            tgt_autoregressive_mask = torch.triu(torch.ones(current_seq_len, current_seq_len, device=x.device) * float('-inf'), diagonal=1)
            # `tgt_padding_mask`: Masks padding tokens if any (not strictly needed here as we build token by token).
            active_tokens_for_masking = current_tgt_tokens[:, :current_seq_len]
            tgt_padding_mask = (active_tokens_for_masking == self.word_vocab['<NULL>'])

            # Embed current target sequences and add positional encoding.
            word_emb = self.vocab_embedding(active_tokens_for_masking) # (batch, current_seq_len, embed_dim)
            word_emb = word_emb.transpose(0, 1) # (current_seq_len, batch, embed_dim)
            word_emb = self.position_encoding(word_emb)

            # Pass through Transformer decoder.
            pred_logits = self.transformer(word_emb, x_visual_features,
                                           tgt_mask=tgt_autoregressive_mask,
                                           tgt_key_padding_mask=tgt_padding_mask) # (current_seq_len, batch, feature_dim)

            # Get logits for the current (last) timestep's prediction.
            current_step_transformer_output = pred_logits[-1, :, :] # (batch, feature_dim)
            current_step_scores = self.wdc(current_step_transformer_output) # (batch, vocab_size)

            # Greedy selection: choose token with highest probability.
            predicted_id = torch.argmax(current_step_scores, dim=-1) # (batch)

            # Update target sequence and generated sequence for each item in the batch.
            all_sequences_ended = True
            for b_idx in range(batch_size):
                # Only append if the sequence for this batch item hasn't ended yet.
                if generated_sequence_ids_batch[b_idx][-1] != self.word_vocab['<END>']:
                    token_to_add = predicted_id[b_idx].item()
                    generated_sequence_ids_batch[b_idx].append(token_to_add)
                    if step < self.max_lengths - 1: # Update input for the next step
                         current_tgt_tokens[b_idx, step + 1] = token_to_add
                    if token_to_add != self.word_vocab['<END>']:
                        all_sequences_ended = False # At least one sequence is still ongoing

            if all_sequences_ended: # Stop if all sequences in batch have generated <END>
                break
        
        # Return the generated sequence for the first item in the batch (assuming batch_size=1 for typical inference).
        return generated_sequence_ids_batch[0]


    def sample1(self, x1: Tensor, x2: Tensor, k: int =1) -> list[int]:
        """
        Beam search sampling method to generate a caption.
        Note: `x1` and `x2` are expected to be image features. This implementation
        concatenates them. If only one input map (e.g. change map) is intended,
        x1 should be that map and x2 could be None or handled accordingly.
        For this refactor, assuming `x1` is the primary input if `x2` is unused by `Conv1`.
        If `Decoder.Conv1` expects `encoder_dim` channels, `torch.cat` is appropriate if `x1` and `x2` sum to that.

        Args:
            x1: First input tensor (e.g., features from image A or a change map).
            x2: Second input tensor (e.g., features from image B or None).
            k: Beam size.

        Returns:
            The highest scoring sequence (list of token IDs) from beam search for the first batch item.
        """
        # Step 0: Prepare visual features (memory for Transformer)
        # This part might need adjustment based on how x1 and x2 are intended to be used.
        # If x1 is the change map (1 channel) and Conv1 expects 1 channel, x2 might be ignored or handled differently.
        # Original code: x = torch.cat([x1, x2], dim = 1)
        # Assuming x1 is the primary input (e.g. change map) and Conv1 handles its `encoder_dim`.
        # If x2 is also part of input, they should be concatenated before Conv1 if Conv1 expects combined channels.
        # For this commenting, let's assume `x_input_to_conv` is correctly formed.
        x_input_to_conv = x1 # Simplified: assuming x1 is the change map and x2 is not used, or Conv1 handles it.
                             # If x1 and x2 must be cat, do: x_input_to_conv = torch.cat([x1, x2], dim=1)

        x_visual_features_processed = self.LN(self.Conv1(x_input_to_conv)) # (batch_size, feature_dim, H, W)
        batch_size_orig = x_visual_features_processed.shape[0] # Original batch size (should be 1 for this beam search impl.)
        if batch_size_orig > 1:
            print("Warning: Beam search sample1 method is designed for batch_size=1. Using first item.")
            x_visual_features_processed = x_visual_features_processed[:1] # Take first item if batch > 1

        channel_dim = x_visual_features_processed.shape[1]
        h_dim, w_dim = x_visual_features_processed.shape[2], x_visual_features_processed.shape[3]

        # Expand visual features for beam size: (H*W, k, feature_dim)
        # Original batch_size is 1 for this beam search. So, effectively (H*W, k, feature_dim).
        x_visual_features = x_visual_features_processed.view(1, channel_dim, -1) # (1, feat, H*W)
        x_visual_features = x_visual_features.expand(k, -1, -1) # (k, feat, H*W) - Incorrect expansion for permute
        # Correct expansion: expand along batch dimension for k beams
        x_visual_features = x_visual_features_processed.view(1, channel_dim, h_dim*w_dim).permute(2,0,1) # (H*W, 1, feat)
        x_visual_features = x_visual_features.expand(-1, k, -1) # (H*W, k, feat)


        # Initialize beam search variables
        # `current_tgt_tokens`: stores the current set of candidate sequences for each beam
        current_tgt_tokens = torch.full((k, self.max_lengths), self.word_vocab['<NULL>'], dtype=torch.long, device=x_input_to_conv.device)
        current_tgt_tokens[:, 0] = self.word_vocab['<START>'] # All k beams start with <START>

        # `generated_sequences`: tracks the full sequences being built for each beam
        generated_sequences = torch.full((k, self.max_lengths), self.word_vocab['<NULL>'], dtype=torch.long, device=x_input_to_conv.device)
        generated_sequences[:, 0] = self.word_vocab['<START>']

        # `top_k_cumulative_scores`: stores the cumulative log-probabilities of the k active beams
        top_k_cumulative_scores = torch.zeros(k, 1, device=x_input_to_conv.device) # Scores for k beams

        # Lists to store completed sequences and their scores
        complete_sequences = []
        complete_sequences_scores = []

        active_beam_count = k # Number of active beams

        # Auto-regressive decoding loop with beam search
        for step in range(self.max_lengths - 1): # Iterate up to max_lengths - 1 tokens
            current_seq_len = step + 1
            if active_beam_count == 0: break # All beams might have ended early

            # Prepare inputs for active beams only
            active_tokens = current_tgt_tokens[:active_beam_count, :current_seq_len]

            # Prepare masks for Transformer
            tgt_autoregressive_mask = torch.triu(torch.ones(current_seq_len, current_seq_len, device=x_input_to_conv.device) * float('-inf'), diagonal=1)
            tgt_padding_mask = (active_tokens == self.word_vocab['<NULL>'])

            # Embed current target sequences and add positional encoding
            word_emb = self.vocab_embedding(active_tokens) # (active_beam_count, current_seq_len, embed_dim)
            word_emb = word_emb.transpose(0, 1) # (current_seq_len, active_beam_count, embed_dim)
            word_emb = self.position_encoding(word_emb)

            # Visual features for active beams: x_visual_features is (H*W, k, feat_dim)
            # We need (H*W, active_beam_count, feat_dim)
            current_visual_features = x_visual_features[:, :active_beam_count, :]

            # Pass through Transformer decoder
            pred_logits = self.transformer(word_emb, current_visual_features,
                                           tgt_mask=tgt_autoregressive_mask,
                                           tgt_key_padding_mask=tgt_padding_mask) # (current_seq_len, active_beam_count, feat_dim)

            current_step_transformer_output = pred_logits[-1, :, :] # (active_beam_count, feat_dim)
            current_step_scores = self.wdc(self.dropout_layer(current_step_transformer_output)) # (active_beam_count, vocab_size)
            current_step_log_probs = F.log_softmax(current_step_scores, dim=1) # (active_beam_count, vocab_size)

            # Add current log-probs to cumulative scores of active beams
            # top_k_cumulative_scores is (active_beam_count, 1)
            scores_expanded = top_k_cumulative_scores[:active_beam_count].expand_as(current_step_log_probs) + current_step_log_probs

            # Select top `active_beam_count` scores and words across all expanded beams.
            # If step=0, we are expanding from 1 initial beam (START token) to k beams.
            # Otherwise, we are expanding from `active_beam_count` beams, each generating `vocab_size` options,
            # and then selecting the top `active_beam_count` overall.
            num_candidates_to_consider = k if step > 0 else active_beam_count # This should always be `active_beam_count` or `k`
                                                                             # to maintain beam width

            # Flatten scores to find global top k candidates
            # Each of the `active_beam_count` beams generates `vocab_size` possibilities
            flattened_scores = scores_expanded.view(-1) # Shape: (active_beam_count * vocab_size)

            # Select top `k` (beam width) overall candidates from these possibilities
            current_top_scores, current_top_indices_flat = flattened_scores.topk(k, 0, True, True)

            # Determine which beam each top-k candidate came from and what the next word is
            prev_beam_indices = torch.div(current_top_indices_flat, self.vocab_size, rounding_mode='floor')
            next_word_indices = current_top_indices_flat % self.vocab_size

            # Reconstruct sequences based on selected beams and next words
            new_generated_sequences = torch.cat([generated_sequences[:active_beam_count][prev_beam_indices], next_word_indices.unsqueeze(1)], dim=1)

            # Identify completed sequences and separate them
            is_next_word_end = (next_word_indices == self.word_vocab['<END>'])
            if step == self.max_lengths - 2: # If at max length, all current beams are considered complete
                 is_next_word_end.fill_(1)

            temp_active_beam_indices = []
            for beam_idx in range(next_word_indices.size(0)): # Iterate through k chosen candidates
                if is_next_word_end[beam_idx]:
                    complete_sequences.append(new_generated_sequences[beam_idx, :current_seq_len+1].tolist())
                    complete_sequences_scores.append(current_top_scores[beam_idx].item())
                else:
                    temp_active_beam_indices.append(beam_idx) # Index within the k candidates

            active_beam_count_new = len(temp_active_beam_indices)

            if active_beam_count_new == 0: break # All beams have ended

            # Update state for active beams for the next iteration
            generated_sequences[:active_beam_count_new] = new_generated_sequences[temp_active_beam_indices]
            # Update visual features if they are dependent on beam state (usually not for fixed encoder output)
            # x_visual_features needs to be re-indexed based on `prev_beam_indices` of the *newly selected active* beams.
            x_visual_features = x_visual_features[:, prev_beam_indices[temp_active_beam_indices]]
            top_k_cumulative_scores[:active_beam_count_new] = current_top_scores[temp_active_beam_indices].unsqueeze(1)

            # Update current_tgt_tokens for the next iteration
            current_tgt_tokens.fill_(self.word_vocab['<NULL>']) # Reset
            current_tgt_tokens[:active_beam_count_new, :current_seq_len+1] = generated_sequences[:active_beam_count_new, :current_seq_len+1]

            active_beam_count = active_beam_count_new # Update number of active beams

        # After loop, if no complete sequences found (e.g., all reached max_len without <END>)
        if not complete_sequences:
            # Add all currently active (but unfinished) beams as completed
            for i in range(active_beam_count):
                complete_sequences.append(generated_sequences[i, :self.max_lengths].tolist()) # Full sequence
                complete_sequences_scores.append(top_k_cumulative_scores[i].item())

        if not complete_sequences_scores: return [self.word_vocab['<START>'], self.word_vocab['<END>']]

        # Select the best sequence among all completed sequences (highest log-probability)
        best_seq_idx = complete_sequences_scores.index(max(complete_sequences_scores))
        best_sequence = complete_sequences[best_seq_idx]

        # Filter out any trailing <NULL> tokens if sequence ended early but was padded
        try:
            first_null = best_sequence.index(self.word_vocab['<NULL>'])
            best_sequence = best_sequence[:first_null]
        except ValueError:
            pass # No <NULL> tokens

        return best_sequence