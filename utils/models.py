from abc import ABC, abstractmethod
import math

import torch
from torch import nn
import torch.nn.functional as F

from .attention import (AdditiveAttention, 
                        PositionalEncoding, 
                        TransformerEncoderBlock, 
                        TransformerDecoderBlock)
from .plot import Plot
from .attention import PatchEmbedding, ViTBlock

Tensor = torch.Tensor

def linear_init(m, hp):
    """
    Initialize the weights of  model.

    Args:
        m (nn.Module): The MLP model to initialize.
        scale (float): Scaling factor for the initialization.
    """
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        hp.init_fn(m.weight, gain=hp.init_scale * nn.init.calculate_gain(hp.activation))
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def scale_init(hp):
    def apply_fn(m):
        return linear_init(m, hp)
    return apply_fn

def get_activation(activation: str):
    if activation == 'relu':
        return F.relu
    elif activation == 'sigmoid':
        return F.sigmoid
    elif activation == 'tanh':
        return F.tanh
    else:
        raise ValueError(f"Unsupported activation function: {activation}")


class Module(Plot, nn.Module):
    def __init__(self):
        super().__init__()
    
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class LogisticRegression(Module):
    def __init__(self, hp):
        super().__init__()
        self.linear = nn.Linear(hp.input_size, hp.num_classes)
        self.apply(scale_init(hp))
    
    def forward(self, x: Tensor) -> Tensor:
        """ 
        Forward pass for the logistic regression model. 

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            Tensor: Output tensor of shape (batch_size, num_classes).
        """
        x = torch.flatten(x, start_dim=1)
        return self.linear(x)


class MLP(Module):
    def __init__(self, hp):
        super().__init__()
        assert len(hp.hidden_widths) > 0, "At least one hidden layer must be specified."
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(hp.input_size, hp.hidden_widths[0]))
        for i in range(len(hp.hidden_widths) - 1):
            self.fc_layers.append(nn.Linear(hp.hidden_widths[i], hp.hidden_widths[i + 1]))
        self.fc_out = nn.Linear(hp.hidden_widths[-1], hp.num_classes)
        self.activation = get_activation(hp.activation)
        self.apply(scale_init(hp))
    
    def forward(self, x: Tensor) -> Tensor:
        """ 
        Forward pass for the MLP model. 

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            Tensor: Output tensor of shape (batch_size, num_classes).
        """
        x = torch.flatten(x, start_dim=1)
        for layer in self.fc_layers:
            x = self.activation(layer(x))
        return self.fc_out(x)

 
class CNN(Module):
    def __init__(self, hp):
        super().__init__()
        assert len(hp.channel_widths) > 0, "At least one convolutional layer must be specified."
        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(hp.input_channels, hp.channel_widths[0], 
                                    kernel_size=3, padding=1))
        for i in range(len(hp.channel_widths) - 1):
            self.convs.append(nn.Conv2d(hp.channel_widths[i], 
                                        hp.channel_widths[i + 1], 
                                        kernel_size=3, padding=1))
        self.fc_hidden = nn.Linear(hp.channel_widths[-1] * (hp.image_width // 
                                    (2 ** min(len(hp.channel_widths), 2))) ** 2, 
                                    hp.hidden_width)
        self.fc_out = nn.Linear(hp.hidden_width, hp.num_classes)
        self.pool = nn.MaxPool2d(kernel_size=hp.pool_size)
        self.activation = get_activation(hp.activation)
        self.batch_norm = hp.batch_norm
        self.apply(scale_init(hp))
        
        if self.batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(len(hp.channel_widths)):
                self.batch_norms.append(nn.BatchNorm2d(hp.channel_widths[i]))
            self.bn_hidden = nn.BatchNorm1d(hp.hidden_width)

    def forward(self, x: Tensor) -> Tensor:
        """ 
        Forward pass for the CNN model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_channels, height, width).
        Returns:
            Tensor: Output tensor of shape (batch_size, num_classes).
        """
        if self.batch_norm:
            for i, (bn, conv) in enumerate(zip(self.batch_norms, self.convs)):
                x = self.activation(bn(conv(x)))
                if i < 2:
                    x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.activation(self.bn_hidden(self.fc_hidden(x)))
        else:
            for i, conv in enumerate(self.convs):
                x = self.activation(conv(x))
                if i < 2:
                    x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.activation(self.fc_hidden(x))
        return self.fc_out(x)

 
def load_model(hp):
    if hp.model_type == 'LogisticRegression':
        model = LogisticRegression(hp)
    elif hp.model_type == 'MLP':
        model = MLP(hp)
    elif hp.model_type == 'CNN':
        model = CNN(hp)
    else:
        raise ValueError(f"Unsupported model type: {hp.model_type}")
    return model


class RecurrentLM(Module, ABC):
    def __init__(self, vocab_size, rnn, out):
        super().__init__()
        self.vocab_size = vocab_size
        self.rnn = rnn
        self.out = out
    
    def forward(self, x, state=None):
        """Performs a forward pass

        Converts the input into a one-hot encoding

        Inputs:
        x (torch.tensor) inputs of shape (batch_size, num_steps)

        Outputs:
        (torch.tensor) logits of shape (batch_size, num_steps, vocab_size)
        """
        x = self.one_hot(x)
        outputs, _ = self.rnn(x, hx=state)
        return self.out(outputs)
    
    def one_hot(self, X):
        return F.one_hot(X, num_classes=self.vocab_size).type(torch.float32)

    def predict(self, prefix, num_preds, vocab):
        state, outputs = None, list(prefix)

        # run model on prefix, get state of model
        X = torch.tensor([vocab[list(prefix)]])
        embs = self.one_hot(X)
        _, state = self.rnn(embs, state)

        # run model forwards one step at a time
        x = torch.tensor([[vocab[prefix[-1]]]])
        emb = self.one_hot(x)
        for i in range(num_preds):
            output, state = self.rnn(emb, state)
            logits = self.out(output)
            pred = torch.argmax(logits)
            outputs.append(vocab.to_tokens(pred.item()))
            emb = self.one_hot(torch.tensor([[pred]]))

        return ''.join(outputs)


class RNNLM(RecurrentLM):
    """RNN-based language model"""
    def __init__(self, vocab_size, hidden_dim):
        rnn = nn.RNN(vocab_size, hidden_dim, batch_first=True)
        linear_out = nn.Linear(hidden_dim, vocab_size)
        super().__init__(vocab_size, rnn, linear_out)


class LSTMLM(RecurrentLM):
    def __init__(self, vocab_size, num_hidden, num_layers=1, 
                 proj_size=0, bidirectional=False):
        rnn = nn.LSTM(vocab_size, num_hidden, batch_first=True,
                            num_layers=num_layers, proj_size=proj_size,
                            bidirectional=bidirectional, )
        hidden_size = (proj_size if proj_size > 0 else num_hidden)
        hidden_size = hidden_size * 2 if bidirectional else hidden_size
        linear_out = nn.Linear(hidden_size, vocab_size)        
        super().__init__(vocab_size, rnn, linear_out)


class Encoder(Module, ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, x, *args):
        pass


class Decoder(Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def init_state(self, enc_all_outputs, *args):
        pass
    
    @abstractmethod
    def forward(self, x, state):
        pass

def init_seq2seq(module):
    """Initialize weights for sequence-to-sequence learning."""
    if type(module) == nn.Linear:
         nn.init.xavier_uniform_(module.weight)
    if type(module) == nn.GRU:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])


class Seq2SeqEncoder(Encoder):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout)
        self.apply(init_seq2seq)
    
    def forward(self, x, *args):
        """Forward pass through encoder network
        
        Inputs:
        
        X (Tensor, torch.int32): (batch_size, num_steps) tokenized src sentences 
        *args: not used

        Outputs:

        outputs (Tensor): (batch_size, num_steps, hidden_size)
        hidden_state (Tensor): (num_layers, batch_size, hidden_size)

        """
        x = self.embed(x)
        outputs, hidden_state = self.rnn(x)
        return outputs, hidden_state


class Seq2SeqDecoder(Decoder):
    def __init__(self, vocab_size, embed_size, hidden_size, 
                 num_layers, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + hidden_size, hidden_size, 
                          num_layers, batch_first=True, dropout=dropout)
        self.linear_out = nn.Linear(hidden_size, vocab_size) 
        self.apply(init_seq2seq)

    def init_state(self, enc_all_outputs, *args):
        """Returns context vector, which is final hidden state
        
        Inputs:
        (outputs, hidden_state) (tuple[Tensor]): hidden states at last layer, final hidden state

        Outputs:
        context (torch.Tensor): (batch_size, hidden_dim) final hidden state
        """
        return enc_all_outputs

    def forward(self, x, init_state):
        """Forward pass through decoder network
        
        Uses final hidden state of encoder network as context vector.
        Appends context vector to tgt token at each timestep.

        Inputs:
        
        X (Tensor, torch.int32): (batch_size, num_steps)
        init_state (torch.Tensor): (batch_size, hidden_size) context vector
        
        Outputs:
        
        outputs (Tensor): (batch_size, num_steps, vocab_size)
        hidden_state (Tensor): (num_layers, batch_size, hidden_size) """
        enc_output, hidden_state = init_state
         # (batchsize, num_steps, embed_dim)
        embed = self.embedding(x)
        # (batch_size, num_steps, hidden_dim
        context = enc_output[:, -1, :]
        context = context.unsqueeze(1).repeat(1, embed.shape[1], 1) 
         # (batch_size, num_steps, hidden_dim + embed_dim)
        embed_and_context = torch.cat((context, embed), dim=2)
        dec_outputs, hidden_state = self.rnn(embed_and_context, hidden_state)
        return self.linear_out(dec_outputs), [enc_output, hidden_state]


class EncoderDecoder(Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, enc_X, dec_X, *args):
        enc_all_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        return self.decoder(dec_X, dec_state)[0]

    def predict_step(self, batch, num_steps, save_attention_weights=False):
        """Unrolls model predictions by taking maximum-probability token

        Sends batch through encoder to create context vector.
        Input to decoder is <bos> token.
        
        Inputs: 
        
        batch (tuple): 
            src (batch_size, num_steps): tokenized input src sentences
            tgt (batch_size, num_steps): tgt sentences, only <bos> used
            src_valid_len (batch_size): padded length of src sentences
            tgt_labels (batch_size, num_steps): tokenized tgt labels, not used
        num_steps (int): number of steps to unroll prediction
        save_attention_weights (bool): used for transformers

        Outputs:

        batch_outputs (torch.Tensor): (batch_size, num_steps) tokenized  
        """
        self.eval()
        src, tgt, src_valid_len, _ = batch
        enc_all_outputs = self.encoder(src, src_valid_len)
        dec_state = self.decoder.init_state(enc_all_outputs, src_valid_len)
        x_dec = tgt[:, 0].unsqueeze(1) # <bos> token
        outputs = [x_dec]
        attention_weights = []
        for _ in range(num_steps):
            output, dec_state = self.decoder(outputs[-1], dec_state)
            outputs.append(output.argmax(2))
            # Save attention weights (to be covered later)
            if save_attention_weights:
                attention_weights.append(self.decoder.attention_weights)
        return torch.cat(outputs[1:], dim=1), attention_weights


class Seq2Seq(EncoderDecoder):
    def __init__(self, encoder, decoder, tgt_pad_idx):
        super().__init__(encoder, decoder)
        self.tgt_pad_idx = tgt_pad_idx


class AttentionDecoder(Decoder):
    def __init__(self):
        super().__init__()
    
    @property
    def attention_weights(self):
        raise NotImplementedError


class Seq2SeqAttentionDecoder(AttentionDecoder):
    """Bahdanau attention decoder"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super().__init__()
        self.attention = AdditiveAttention(num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers, 
            batch_first=True, dropout=dropout)
        self.dense = nn.LazyLinear(vocab_size)
        self.apply(init_seq2seq)

    def init_state(self, all_enc_outputs, enc_valid_lens):
        """Returns context vector, which is final hidden state
        
        Inputs:
        (outputs, hidden_state) (tuple[Tensor]): 
            enc_outputs: (B, N, H) encoder outputs 
            hidden_state: (L, B, H) final encoder hidden state
        enc_valid_lens (Tensor): (B) number of non-padded tokens

        Outputs:
        (enc_outputs, hidden_state, valid_lens) (tuple[Tensor])
        """
        enc_outputs, hidden_state = all_enc_outputs
        return enc_outputs, hidden_state, enc_valid_lens

    def forward(self, X, init_state):
        """Forward pass through decoder

        B: batch size
        N: num_steps, length of tgt sentences
        H: num_hiddens
        E: embed_size
        L: num_layers
        V: target vocab size

        Args:
            X (torch.Tensor): (B, N) tgt sentences, indices
            init_state (tuple): enc_outputs, hidden_state, enc_valid_lens
        
        Returns:
            torch.Tensor: (B, N, H) output of decoder
            tuple: next state of encoder, only hidden_state changes
        """
        enc_outputs, hidden_state, enc_valid_lens = init_state
        # (B, N, E) -> (N, B, E), allows iteration over X
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            # Gives s_{t'-1}, (B, 1, H)
            query = hidden_state[-1].unsqueeze(dim=1)
            # Bahdanau context c_t' = \sum \alpha(s_{t'-1}, h_t) h_t
            context = self.attention(
                query, enc_outputs, enc_outputs, valid_lens=enc_valid_lens)
            # (B, 1, H + E)
            x = torch.cat((context, x.unsqueeze(dim=1)), dim=-1)
             # this only works if num_layers in encoder is same as decoder
             # out is (B, 1, H), hidden_state is (L, B, H)
            out, hidden_state = self.rnn(x, hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # (B, N, H)
        outputs = torch.cat(outputs, dim=1)
        # (B, N, V)
        outputs = self.dense(outputs)
        return outputs, [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights


class TransformerEncoder(Encoder):
    """Encoder for encoder-decoder transformer
    
    For details, see:
    https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html 
    
    Positional encoding is added to the input embeddings, and then
    num_blks TransformerEncoderBlocks are applied sequentially.
    """
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, 
                 num_heads, num_blks, dropout, use_bias=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block"+str(i), TransformerEncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias))
    
    def forward(self, X, valid_lens):
        X = self.embedding(X)
        X = self.pos_encoding(X * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X


class TransformerDecoder(Decoder):
    """Decoder for encoder-decoder transformer
    
    For details, see:
    https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html
    
    Positional encoding is added to the input embeddings, and then
    num_blks TransformerDecoderBlocks are applied sequentially."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads, 
                 num_blks, dropout):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_blks = num_blks
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block"+str(i), TransformerDecoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, i
            ))
        self.dense = nn.LazyLinear(vocab_size)
    
    def init_state(self, enc_outputs, enc_valid_lens):
        return enc_outputs, enc_valid_lens, [None] * self.num_blks

    def forward(self, X, state):
        X = self.embedding(X)
        X = self.pos_encoding(X * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights


class ViT(Module):
    """Vision transformer."""
    def __init__(self, img_size, patch_size, num_hiddens, mlp_num_hiddens,
                 num_heads, num_blks, emb_dropout, blk_dropout, 
                 use_bias=False, num_classes=10):
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            img_size, patch_size, num_hiddens
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_hiddens))
        num_steps = self.patch_embedding.num_patches + 1
        self.pos_embedding = nn.Parameter(torch.randn(1, num_steps, num_hiddens))
        self.dropout = nn.Dropout(emb_dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            blk = ViTBlock(
                num_hiddens, mlp_num_hiddens, num_heads, blk_dropout, use_bias
            )
            self.blks.add_module(f"block {i}", blk)
        self.head = nn.Sequential(nn.LayerNorm(num_hiddens), 
                                  nn.Linear(num_hiddens, num_classes))
    
    def forward(self, X):
        X = self.patch_embedding(X)
        cls_tokens = self.cls_token.expand(X.shape[0], -1, -1)
        X = torch.cat((cls_tokens, X), dim=1)
        X = self.dropout(X + self.pos_embedding)
        for blk in self.blks:
            X = blk(X)
        return self.head(X[:, 0])