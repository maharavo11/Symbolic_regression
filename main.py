# Main script to run the symbolic regression pipeline.

import torch
import numpy as np
from data_generator import generate_and_filter_dataset
from symbolic_regression_utils import create_token_maps, tree_to_prefix_sequence
from models import SeqGRU
from trainer import train_model, evaluate_model
from config import *

def main():
    # 1. Generate and Filter Dataset
    dataset = generate_and_filter_dataset(NUM_EXPRESSIONS_TO_GENERATE)
    
    # 2. Tokenization
    forward_token_map, backward_token_map = create_token_maps(BINARY_OPERATORS, UNARY_OPERATORS)
    token_maps = {'forward': forward_token_map, 'backward': backward_token_map}
    n_tokens = len(forward_token_map)
    
    vecs = [
        [forward_token_map[s] for s in tree_to_prefix_sequence(item['root'])]
        for item in dataset
    ]
    datas = np.array([item['data'] for item in dataset])
    
    # 3. Prepare Data for PyTorch
    max_len = max(len(v) for v in vecs) + 1  # For <EOS>
    
    vecs_padded = np.array([
        v + [forward_token_map["<EOS>"]] + [forward_token_map["<PAD>"]] * (max_len - len(v) - 1)
        for v in vecs
    ])

    split_idx = int(TRAIN_TEST_SPLIT * len(datas))
    
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(datas[:split_idx], dtype=torch.float32),
        torch.tensor(vecs_padded[:split_idx], dtype=torch.long)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(datas[split_idx:], dtype=torch.float32),
        torch.tensor(vecs_padded[split_idx:], dtype=torch.long)
    )
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"\nDataset prepared. Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    
    # 4. Initialize and Train Model
    model = SeqGRU(
        N=NUM_DATA_POINTS, T=max_len, n_tokens=n_tokens,
        d_model=D_MODEL, n_enc_layers=N_ENC_LAYERS,
        dec_hidden_dim=DEC_HIDDEN_DIM, n_gru_layers=N_GRU_LAYERS
    )
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):.2e} parameters.")
    
    trained_model = train_model(model, train_loader, test_loader, NUM_TRAINING_STEPS, token_maps)
    
    # 5. Final Evaluation
    print("\n--- Final Evaluation ---")
    evaluate_model(trained_model, test_loader, "Final Test", token_maps)

if __name__ == "__main__":
    main()