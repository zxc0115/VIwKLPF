import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.init import xavier_normal
from torch.nn.parameter import Parameter
import flows
class linIAF(nn.Module):
    def __init__(self):
        super(linIAF, self).__init__()

    def forward(self, L, z, z_dim):
        '''
        :param L: batch_size (B) x latent_size^2 (L^2)
        :param z: batch_size (B) x latent_size (L)
        :return: z_new = L*z
        '''
        # L->tril(L)
        L_matrix = L.view( -1, z_dim, z_dim ) # resize to get B x L x L
        LTmask = torch.tril( torch.ones(z_dim, z_dim), diagonal=-1 ) # lower-triangular mask matrix (1s in lower triangular part)
        I = torch.Tensor( torch.eye(z_dim, z_dim).expand(L_matrix.size(0), z_dim, z_dim) )
        if torch.cuda.is_available():
            LTmask = LTmask.cuda()
            I = I.cuda()
        #LTmask = torch.Tensor(LTmask)
        LTmask = LTmask.unsqueeze(0).expand( L_matrix.size(0), z_dim, z_dim ) # 1 x L x L -> B x L x L
        LT = torch.mul( L_matrix, LTmask ) + I # here we get a batch of lower-triangular matrices with ones on diagonal

        # z_new = L * z
        z_new = torch.bmm( LT , z.unsqueeze(2) ).squeeze(2) # B x L x L * B x L x 1 -> B x L

        return z_new

class combination_L(nn.Module):
    def __init__(self):
        super(combination_L, self).__init__()

    def forward(self, L, y, n_combi, z_dim):
        '''
        :param L: batch_size (B) x latent_size^2 * number_combination (L^2 * C)
        :param y: batch_size (B) x number_combination (C)
        :return: L_combination = y * L
        '''
        # calculate combination of Ls
        L_tensor = L.view( -1, z_dim**2, n_combi ) # resize to get B x L^2 x C
        y = y.unsqueeze(1).expand(y.size(0), z_dim**2, y.size(1)) # expand to get B x L^2 x C
        L_combination = torch.sum( L_tensor * y, 2 ).squeeze()
        return L_combination



class iafEncoder(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, z_dim, pad_idx):
        super(iafEncoder, self).__init__()
        # input
        self.embedding = nn.Embedding(vocab_size, embed_size,
                                      padding_idx=pad_idx)
        # RNN
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)
        # output
        self.output = nn.Linear(hidden_size * 2, z_dim * 2)
        # iaf highway
        self.highway = nn.Linear(hidden_size * 2, hidden_size)
        

    def forward(self, input_seq, length):
        # embed input
        embedded_input = self.embedding(input_seq)
        embedded_input = embedded_input + torch.randn_like(embedded_input)

        # RNN forward
        pack_input = pack_padded_sequence(embedded_input, length,
                                          batch_first=True)
        _, (h, c) = self.rnn(pack_input)

        # produce mu and logvar
        hidden = torch.cat([h, c], dim=-1).squeeze(0)
        mu, logvar = torch.chunk(self.output(hidden), 2, dim=-1)
        highway = self.highway(hidden)

        return mu, logvar, highway

class KLPF(nn.Module):

    def __init__(self, vocab_size, embed_size, time_step, hidden_size, z_dim,
                 dropout_rate, bos_idx, eos_idx, pad_idx, n_comb):
        super(KLPF, self).__init__()
        self.time_step = time_step
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.number_combination = n_comb
        self.z_dim = z_dim

        # encoder
        self.encoder = iafEncoder(vocab_size, embed_size,
                                   hidden_size, z_dim, pad_idx)
        # iaf
        #self.linIAF = linIAF()
        #self.combination_L = combination_L()
        #self.encoder_y = nn.Linear( hidden_size, self.number_combination )
        #self.encoder_L = nn.Linear( hidden_size, (z_dim**2) * self.number_combination )
        self.flow = flows.myIAF(z_dim, z_dim, hidden_size,1)
        self.softmax = nn.Softmax()
        
        # decoder
        self.embedding = nn.Embedding(vocab_size, embed_size,
                                      padding_idx=pad_idx)
        self.init_h = nn.Linear(z_dim, hidden_size)
        self.init_c = nn.Linear(z_dim, hidden_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)
        # output
        self.output = nn.Linear(hidden_size, vocab_size)
        
    def q_z_Flow(self, z_0, weight, h_last, mu):
        # ccLinIAF
        #L = self.encoder_L(h_last)
        #y = self.softmax(self.encoder_y(h_last))
        #L_combination = self.combination_L(L, y, self.number_combination, self.z_dim)
        #z = self.linIAF(L_combination, z_0, self.z_dim)
        z , _, _, _ = self.flow([z_0,weight,h_last, mu])
        return z

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + std * eps

    def forward(self, enc_input, dec_input, length, labels):
        # process input
        sorted_len, sorted_idx = torch.sort(length, descending=True)
        enc_input = enc_input[sorted_idx]
        dec_input = dec_input[sorted_idx]
        labels = labels[sorted_idx]
        
        # encode
        mu, logvar, h = self.encoder(enc_input, sorted_len)
        z_0 = self.reparameterize(mu, logvar)
        weight = torch.eye(z_0.shape[1])
        z_T = self.q_z_Flow(z_0, weight, h, mu)

        # decode
        embedded_input = self.embedding(dec_input)
        drop_input = F.dropout(embedded_input, p=self.dropout_rate,
                               training=self.training)
        pack_input = pack_padded_sequence(drop_input, sorted_len + 1,
                                          batch_first=True)
        h_0, c_0 = self.init_h(z_T), self.init_c(z_T)
        hidden = (h_0.unsqueeze(0), c_0.unsqueeze(0))
        pack_output, _ = self.rnn(pack_input, hidden)
        output, _ = pad_packed_sequence(pack_output, batch_first=True)
        _, reversed_idx = torch.sort(sorted_idx)
        output = output[reversed_idx]

        # project output
        batch_size, seq_len, hidden_size = output.size()
        logit = self.output(output.view(-1, hidden_size))
        logp = F.log_softmax(logit, dim=-1)
        logp = logp.view(batch_size, seq_len, -1)

        return logp, z_0, z_T, mu, logvar, labels

    def inference(self, z):
        # set device
        tensor = torch.LongTensor
        if torch.cuda.is_available():
            tensor = torch.cuda.LongTensor

        # initialize hidden state
        batch_size = z.size(0)
        h_0, c_0 = self.init_h(z), self.init_c(z)
        hidden = (h_0.unsqueeze(0), c_0.unsqueeze(0))

        # RNN forward
        symbol = tensor(batch_size, self.time_step + 1).fill_(self.pad_idx)
        for t in range(self.time_step + 1):
            if t == 0:
                input_seq = tensor(batch_size, 1).fill_(self.bos_idx)
            embedded_input = self.embedding(input_seq)
            output, hidden = self.rnn(embedded_input, hidden)
            logit = self.output(output)
            _, sample = torch.topk(logit, 1, dim=-1)
            input_seq = sample.squeeze(-1)
            symbol[:, t] = input_seq.squeeze(-1)

        return symbol
    
