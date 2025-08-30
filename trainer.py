# Training and evaluation loops for the model.

import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from config import LEARNING_RATE, DEVICE
from plotter import plot_predictions

def train_model(model, train_loader, test_loader, n_steps, token_maps):
    """Main training loop."""
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=token_maps['forward']['<PAD>'])
    
    step = 0
    pbar = tqdm(total=n_steps, desc="Training")
    
    while step < n_steps:
        for data_batch, vecs_batch in train_loader:
            if step >= n_steps:
                break
            
            model.train()
            data_batch, vecs_batch = data_batch.to(DEVICE), vecs_batch.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Teacher forcing
            logits = model(data_batch, vecs_batch[:, :-1])
            loss = criterion(logits.reshape(-1, logits.shape[-1]), vecs_batch[:, 1:].reshape(-1))
            
            loss.backward()
            optimizer.step()
            
            pbar.set_description(f"Step {step}/{n_steps}, Loss: {loss.item():.4f}")
            pbar.update(1)
            
            if step > 0 and step % (n_steps // 5) == 0:
                evaluate_model(model, test_loader, "Test", token_maps)
                
            step += 1
            
    pbar.close()
    return model

def evaluate_model(model, loader, tag, token_maps):
    """Evaluate model accuracy."""
    model.eval()
    total_correct = 0
    total_tokens = 0
    pad_token_idx = token_maps['forward']['<PAD>']
    eos_token_idx = token_maps['forward']['<EOS>']

    with torch.no_grad():
        for data_batch, vecs_batch in loader:
            data_batch, vecs_batch = data_batch.to(DEVICE), vecs_batch.to(DEVICE)
            
            # Generate predictions up to the length of the target
            max_len = vecs_batch.size(1)
            preds = model.inference(data_batch, start_token_idx=pad_token_idx, eos_token_idx=eos_token_idx)
            
            # Ensure preds and targets have same length for comparison
            if preds.size(1) < max_len:
                padding = torch.full((preds.size(0), max_len - preds.size(1)), pad_token_idx, device=DEVICE)
                preds = torch.cat([preds, padding], dim=1)
            
            mask = (vecs_batch != pad_token_idx)
            total_correct += ((preds[:, :max_len] == vecs_batch) & mask).sum().item()
            total_tokens += mask.sum().item()
            
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    print(f"\nAccuracy on {tag} dataset: {accuracy:.4f}")
    
    # Plot some predictions
    plot_predictions(model, loader, token_maps)