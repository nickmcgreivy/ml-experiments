from typing import List, Callable, Tuple
from abc import ABC, abstractmethod

import torch
from torch import nn
import torch.nn.functional as F

from .plot import Plot

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
                                    (2 ** (len(hp.channel_widths) - 1))) ** 2, 
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
                if i > 0:
                    x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.activation(self.bn_hidden(self.fc_hidden(x)))
        else:
            for i, conv in enumerate(self.convs):
                x = self.activation(conv(x))
                if i > 0:
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
        src, tgt, src_valid_len, _ = batch
        enc_all_outputs = self.encoder(src, src_valid_len)
        context = self.decoder.init_state(enc_all_outputs, src_valid_len)
        x_dec = tgt[:, 0].unsqueeze(1) # <bos> token
        outputs = [x_dec]
        attention_weights = []
        for _ in range(num_steps):
            output, hidden_state = self.decoder(outputs[-1], context)
            outputs.append(output.argmax(2))
            # Save attention weights (to be covered later)
            if save_attention_weights:
                attention_weights.append(self.decoder.attention_weights)
        return torch.cat(outputs[1:], dim=1), attention_weights

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
        self.linear_out= nn.Linear(hidden_size, vocab_size) 
        self.apply(init_seq2seq)

    def init_state(self, enc_all_outputs, *args):
        """Returns context vector, which is final hidden state
        
        Inputs:
        (outputs, hidden_state) (tuple[Tensor]): hidden states at last layer, final hidden state

        Outputs:
        context (torch.Tensor): (batch_size, hidden_dim) final hidden state
        """
        outputs, _ = enc_all_outputs
        return outputs[:, -1, :]

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
         # (batchsize, num_steps, embed_dim)
        embed = self.embedding(x)
        # (batch_size, num_steps, hidden_dim
        context = init_state.unsqueeze(1).repeat(1, embed.shape[1], 1) 
         # (batch_size, num_steps, hidden_dim + embed_dim)
        embed_and_context = torch.cat((context, embed), dim=2)
        dec_outputs, hidden_state = self.rnn(embed_and_context)
        return self.linear_out(dec_outputs), hidden_state

class Seq2Seq(EncoderDecoder):
    def __init__(self, encoder, decoder, tgt_pad_idx):
        super().__init__(encoder, decoder)
        self.tgt_pad_idx = tgt_pad_idx