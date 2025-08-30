# PyTorch models for symbolic regression.

import torch
import torch.nn as nn

class SeqGRU(nn.Module):
    """A GRU-based encoder-decoder for symbolic regression."""
    def __init__(self, N, T, n_tokens, d_model, n_enc_layers, dec_hidden_dim, n_gru_layers):
        super().__init__()
        self.T = T
        self.token_embed = nn.Embedding(n_tokens, d_model)

        enc_layers = [nn.Linear(N * 2, d_model), nn.ReLU()]
        for _ in range(n_enc_layers - 1):
            enc_layers.extend([nn.Linear(d_model, d_model), nn.ReLU()])
        self.encoder = nn.Sequential(*enc_layers)

        self.gru = nn.GRU(
            input_size=d_model, hidden_size=dec_hidden_dim,
            num_layers=n_gru_layers, batch_first=True
        )
        self.out_proj = nn.Linear(dec_hidden_dim, n_tokens)

    def forward(self, data, x):
        """Forward pass for teacher-forced training."""
        B = data.size(0)
        data_encoded = self.encoder(data.view(B, -1))
        h0 = data_encoded.unsqueeze(0).repeat(self.gru.num_layers, 1, 1)
        x_embedded = self.token_embed(x)
        out, _ = self.gru(x_embedded, h0)
        logits = self.out_proj(out)
        return logits

    def inference(self, data, start_token_idx, eos_token_idx):
        """Autoregressive generation for inference."""
        B = data.size(0)
        device = data.device
        
        data_encoded = self.encoder(data.view(B, -1))
        h = data_encoded.unsqueeze(0).repeat(self.gru.num_layers, 1, 1)
        
        # Start with the PAD token as the initial input
        x = torch.full((B, 1), start_token_idx, dtype=torch.long, device=device)
        
        output_tokens = []
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(self.T):
            x_embedded = self.token_embed(x[:, -1]).unsqueeze(1)
            out, h = self.gru(x_embedded, h)
            logits = self.out_proj(out.squeeze(1))
            next_token = torch.argmax(logits, dim=-1)
            
            # Stop generating for sequences that have hit EOS
            next_token[finished] = eos_token_idx
            finished |= (next_token == eos_token_idx)

            output_tokens.append(next_token.unsqueeze(1))
            x = torch.cat([x, next_token.unsqueeze(1)], dim=1)

            if finished.all():
                break

        return torch.cat(output_tokens, dim=1)