import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import json

with open('D:/PolyU/URIS/Part2_projects/WEMOM_V1/EmoText/TextVAE_config.json') as f:
    args = json.load(f)

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.rnn = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = dropout

    def forward(self, text):
        text_mask = (text != args["PAD_INDEX"])
        text_lens = text_mask.long().sum(dim=1, keepdim=False)
        text_embedding = self.embedding(text)
        text = F.dropout(text_embedding, p=self.dropout, training=self.training)

        text_lens, sort_index = text_lens.sort(descending=True)
        text = text.index_select(dim=0, index=sort_index)
        text_lens = text_lens.cpu()

        packed_text = pack_padded_sequence(text, text_lens, batch_first=True)
        packed_output, final_states = self.rnn(packed_text)

        reorder_index = sort_index.argsort(descending=False)
        final_states = final_states.index_select(dim=1, index=reorder_index)

        num_layers = self.rnn.num_layers
        num_directions = 2 if self.rnn.bidirectional else 1
        final_states = final_states.view(num_layers, num_directions, -1, self.rnn.hidden_size)
        final_states = final_states[-1]
        final_states = torch.cat([final_states[0], final_states[1]], dim=1)

        return final_states

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.rnn = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, target=None, max_len=50):
        num_layers = self.rnn.num_layers
        hidden = hidden.unsqueeze(0).repeat(num_layers, 1, 1)

        if target is not None:  # Teacher forcing
            token_embedding = self.embedding(target)
            output, _ = self.rnn(token_embedding, hidden)
            logits = self.output_projection(output)
        else:  # Autoregressive decoding
            batch_size = hidden.size(1)
            token = torch.tensor([args["SOS_INDEX"]] * batch_size, dtype=torch.long, device=hidden.device)
            logits = []
            for _ in range(max_len):
                token_embedding = self.embedding(token).unsqueeze(1)
                output, hidden = self.rnn(token_embedding, hidden)
                token_logit = self.output_projection(output.squeeze(1))
                token = token_logit.argmax(dim=-1)
                logits.append(token_logit)
            logits = torch.stack(logits, dim=1)

        return logits

class TextVAE(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout, num_labels=4):
        super(TextVAE, self).__init__()
        self.encoder = Encoder(vocab_size, embed_size, hidden_size, num_layers, dropout)
        self.decoder = Decoder(vocab_size, embed_size, hidden_size, num_layers, dropout)
        self.mean_projection = nn.Linear(2 * hidden_size, hidden_size)
        self.std_projection = nn.Linear(2 * hidden_size, hidden_size)
        self.decoder_projection = nn.Linear(vocab_size, hidden_size)
        self.classifier = nn.Linear(hidden_size*2, num_labels)  

    def forward(self, text):
        text_input = text[:, :-1]  # Remove <eos>
        target_input = text[:, :-1]  # Remove <eos>
        encoding, mean, std = self.encode(text_input)
        decoder_output = self.decoder(encoding, target_input)
        decoder_features = self.decoder_projection(decoder_output.mean(dim=1, keepdim=True))  # 保留维度
        combined_features = torch.cat([encoding, decoder_features.squeeze(1)], dim=1)  # 拼接
        logits = self.classifier(combined_features)
        return decoder_output, mean, std, logits

    def encode(self, text):
        final_states = self.encoder(text)
        mean = self.mean_projection(final_states)
        std = F.softplus(self.std_projection(final_states))
        sample = torch.randn(size=mean.size(), device=mean.device)
        encoding = mean + std * sample
        return encoding, mean, std