"""
the implementations are from https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Initialize dimensions
        self.d_model = d_model  # Model's dimension
        self.num_heads = num_heads  # Number of attention heads
        self.d_k = (
            d_model // num_heads
        )  # Dimension of each head's key, query, and value

        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model)  # Query transformation
        self.W_k = nn.Linear(d_model, d_model)  # Key transformation
        self.W_v = nn.Linear(d_model, d_model)  # Value transformation
        self.W_o = nn.Linear(d_model, d_model)  # Output transformation

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        # batch_size, num_heads, seq_length, d_k
        # matmul by default performs on the last 2 dimensions, so Q(seq_length, dk) @ K (dk, seq_length) -> score(batch_size, num_heads, seq_length, seq_length)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Softmax is applied to obtain attention probabilities
        # batch_size, num_heads, seq_length, seq_length
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Multiply by values to obtain the final output
        # batch_size, num_heads, seq_length, seq_length @ atch_size, num_heads, seq_length, d_k = atch_size, num_heads, seq_length, d_k 
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        # d_model = num_heads * d_k
        # output dim batch_size, num_heads, seq_length, d_k
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        # here we make x in dim batch_size, seq_length, n_heads, d_k, -> batch_size, seq_length, d_model
        # he .contiguous() method in PyTorch ensures that the tensor's memory layout is contiguous in the expected order. This step is crucial because the subsequent .view() operation relies on a contiguous memory block to correctly reshape the data.
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        weights_Q = self.W_q(Q)
        Q = self.split_heads(weights_Q)
        # Q in dim: batch_size, num_heads, seq_length, d_k 
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # Perform scaled dot-product attention
        # output in atch_size, num_heads, seq_length, d_k
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads and apply output transformation
        # after combine_heads, it outputs (batch_size, seq_length, d_model)
        # applying W_o will do X@W_o=B, in dim (batch_size, seq_length, d_model) @ (d_model, d_model) + d_model = (batch_size, seq_length, d_model)
        # it is doing X@W+b
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        """
        
        :param d_model: 
        :param max_seq_length: with max_seq_length as input, we are sure that the pe has the same dimension as the input src 
        """
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register pe as a buffer to avoid it being considered a model parameter
        # unsqueeze to add batch dimension -> batch_size x max_seq_length x d_model
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """
        
        :param x: shape batch, max_seq_length, d_model 
        :return: 
        """
        return x + self.pe[:, :x.size(1), :]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout,
    ):
        super(Transformer, self).__init__()
        # in practice the src_vocab_size is determined according to the language and training set
        # 1. we choose a tokenizer, e.g., Byte-Pair Encoding (BPE), WordPiece, or SentencePiece.
        # 2. we set a target vocab size, e.g., 30-60k for monoligual English
        # 3. build the vocabulary using our training set and target vocab size
        # 4. final vocab_size = target vocab size + special tokens (e.g., [PAD], [UNK], [CLS])
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        """
        This method is used to create masks for the source and target sequences, ensuring that padding tokens are ignored and that future tokens are not visible during training for the target sequence.
        :param src: batch_size, seq_length, in practice the seq_length is the maximum length of the seqs in a batch, if a seq is smaller, we pad zeros before sending to here, so here we need to mask those zero paddings.
        :param tgt: 
        :return: 
        """
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (
            1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)
        ).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        # 16, 1, 1, 20, and 16, 1, 99, 99
        src_raw_embedding = self.encoder_embedding(src)
        # 16, 20, 512 == batch, seq_length, d_model
        src_after_pe = self.positional_encoding(src_raw_embedding)
        # in original paper: Residual Dropout We apply dropout [33] to the output of each sub-layer, before it is added to the
        # sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the
        # positional encodi
        src_embedded = self.dropout(src_after_pe)
        tgt_embedded = self.dropout(
            self.positional_encoding(self.decoder_embedding(tgt))
        )
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output


if __name__ == "__main__":
    """
    src_vocab_size: Source vocabulary size.
    tgt_vocab_size: Target vocabulary size.
    d_model: The dimensionality of the model's embeddings.
    num_heads: Number of attention heads in the multi-head attention mechanism.
    num_layers: Number of layers for both the encoder and the decoder.
    d_ff: Dimensionality of the inner layer in the feed-forward network.
    max_seq_length: Maximum sequence length for positional encoding.
    dropout: Dropout rate for regularization.
    """
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model = 512
    num_heads = 8
    # d_model // num_heads must be zero because we end up with concate output of each head to have the d_model length at the end, so num_heads must be dividable by d_model
    num_layers = 6
    d_ff = 2048
    max_seq_length = 100
    dropout = 0.1

    transformer = Transformer(
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout,
    )

    # Generate random sample data
    src_data = torch.randint(
        1, src_vocab_size, (16, 20)
    )  # (batch_size, seq_length)
    # for each batch of batch_size of sequencies, in practice, the dataloader should be responsible to align the sequence length by
    # 1. find the maximum seq length in this batch
    # 2. for each sequence in this batch, fill zeros if it is smaller than the maximum seq length
    # 3. finally all sequencies will have the same length -> seq_length

    tgt_data = torch.randint(
        1, tgt_vocab_size, (16, max_seq_length)
    )  # (batch_size, seq_length)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    transformer.train()

    for epoch in range(100):
        optimizer.zero_grad()
        output = transformer(src_data, tgt_data[:, :-1])
        loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

    transformer.eval()

    # Generate random sample validation data
    val_src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
    val_tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

    with torch.no_grad():

        val_output = transformer(val_src_data, val_tgt_data[:, :-1])
        val_loss = criterion(val_output.contiguous().view(-1, tgt_vocab_size),
                             val_tgt_data[:, 1:].contiguous().view(-1))
        print(f"Validation Loss: {val_loss.item()}")
