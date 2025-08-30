# plotter.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sympy as sp
from symbolic_regression_utils import prefix_sequence_to_tree, tree_to_sympy
from config import (
    NUM_PREDICTION_PLOTS, OUTPUT_DIR, BINARY_OPERATORS, UNARY_OPERATORS,
    X_RANGE_MIN, X_RANGE_MAX, NUM_DATA_POINTS, FILTER_Y_MAX_ABS_VALUE  # <-- Added here
)

def plot_predictions(model, loader, token_maps):
    """Plots model predictions against true functions."""
    model.eval()
    data_batch, vecs_batch = next(iter(loader))
    
    with torch.no_grad():
        preds_vecs = model.inference(
            data_batch.to(next(model.parameters()).device),
            start_token_idx=token_maps['forward']['<PAD>'],
            eos_token_idx=token_maps['forward']['<EOS>']
        )
    
    num_plots = min(NUM_PREDICTION_PLOTS, len(data_batch))
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()

    for i in range(num_plots):
        ax = axes[i]
        
        # True expression
        true_seq = [token_maps['backward'][t] for t in vecs_batch[i].tolist() if t != token_maps['forward']['<PAD>']]
        if "<EOS>" in true_seq: true_seq = true_seq[:true_seq.index("<EOS>")]
        
        # Predicted expression
        pred_seq = [token_maps['backward'][t] for t in preds_vecs[i].tolist()]
        if "<EOS>" in pred_seq: pred_seq = pred_seq[:pred_seq.index("<EOS>")]

        try:
            true_expr = tree_to_sympy(prefix_sequence_to_tree(true_seq, BINARY_OPERATORS, UNARY_OPERATORS), BINARY_OPERATORS, UNARY_OPERATORS)
        except Exception:
            true_expr = "Invalid"
            
        try:
            pred_expr = tree_to_sympy(prefix_sequence_to_tree(pred_seq, BINARY_OPERATORS, UNARY_OPERATORS), BINARY_OPERATORS, UNARY_OPERATORS)
        except Exception:
            pred_expr = "Invalid"
            
        ax.scatter(data_batch[i, :, 0], data_batch[i, :, 1], label='Data', s=10)
        x_plot = np.linspace(X_RANGE_MIN, X_RANGE_MAX, NUM_DATA_POINTS)
        
        if true_expr != "Invalid":
            fn_true = sp.lambdify(sp.Symbol('x'), true_expr, 'numpy')
            ax.plot(x_plot, fn_true(x_plot), label=f'True: {true_expr}', c='blue')
        
        if pred_expr != "Invalid":
            fn_pred = sp.lambdify(sp.Symbol('x'), pred_expr, 'numpy')
            ax.plot(x_plot, fn_pred(x_plot), label=f'Pred: {pred_expr}', c='orange', ls='--')
            
        ax.set_title(f"Sample {i+1}")
        ax.legend(fontsize='x-small')
        # This line now works correctly
        ax.set_ylim(-FILTER_Y_MAX_ABS_VALUE, FILTER_Y_MAX_ABS_VALUE)

    plt.tight_layout()
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    plt.savefig(os.path.join(OUTPUT_DIR, "final_predictions.png"))
    print(f"\nSaved prediction plots to {os.path.join(OUTPUT_DIR, 'final_predictions.png')}")
    plt.close()