# Functions for generating and filtering the expression dataset.

import random
import numpy as np
import sympy as sp
from tqdm import tqdm
from symbolic_regression_utils import TreeNode, str_tree, tree_to_sympy
from config import (
    BINARY_OPERATORS, UNARY_OPERATORS, LEAVES, MAX_BINARY_OPS,
    MAX_UNARY_OPS, NUM_DATA_POINTS, X_RANGE_MIN, X_RANGE_MAX,
    FILTER_Y_MAX_ABS_VALUE
)

def _generate_random_binary_tree(n_operators):
    """Helper to generate a random binary tree."""
    if n_operators == 0:
        return random.choice(LEAVES)

    root = TreeNode(val=random.choice(list(BINARY_OPERATORS.keys())))
    n_operators -= 1
    
    left_n = random.randint(0, n_operators)
    right_n = n_operators - left_n
    
    root.left = _generate_random_binary_tree(left_n)
    root.right = _generate_random_binary_tree(right_n)
    
    return root

def _add_random_unary_operators(root, n_operators):
    """Helper to add unary operators to a tree."""
    if not isinstance(root, TreeNode) or n_operators == 0:
        return root

    nodes = []
    def collect_nodes(node):
        if isinstance(node, TreeNode):
            nodes.append(node)
            collect_nodes(node.left)
            collect_nodes(node.right)
    collect_nodes(root)

    for _ in range(n_operators):
        if not nodes: break
        chosen_node = random.choice(nodes)
        new_node = TreeNode(
            val=random.choice(list(UNARY_OPERATORS.keys())),
            left=chosen_node, right=None
        )
        if root is chosen_node:
            root = new_node
        else:
            for node in nodes:
                if node.left is chosen_node:
                    node.left = new_node; break
                elif node.right is chosen_node:
                    node.right = new_node; break
        nodes.append(new_node)
    return root

def generate_and_filter_dataset(num_expressions):
    """Main function to generate, evaluate, and filter the dataset."""
    print(f"Generating and filtering dataset of ~{num_expressions} expressions...")
    
    # 1. Generate Raw Expressions
    roots, exprs = [], []
    for _ in tqdm(range(num_expressions), desc="Generating expressions"):
        n_binary = random.randint(1, MAX_BINARY_OPS)
        n_unary = random.randint(0, MAX_UNARY_OPS)
        tree = _generate_random_binary_tree(n_binary)
        tree = _add_random_unary_operators(tree, n_unary)
        roots.append(tree)
        exprs.append(tree_to_sympy(tree, BINARY_OPERATORS, UNARY_OPERATORS))

    # 2. Evaluate and Filter Numerically
    x_values = np.linspace(X_RANGE_MIN, X_RANGE_MAX, NUM_DATA_POINTS)
    x_sym = sp.Symbol('x')
    
    valid_data = []
    for root, expr in tqdm(zip(roots, exprs), total=len(roots), desc="Evaluating expressions"):
        try:
            if isinstance(expr, sp.Integer):
                y_values = np.full_like(x_values, float(expr))
            else:
                fn = sp.lambdify(args=x_sym, expr=expr, modules="numpy")
                y_values = fn(x_values)
            
            if np.isscalar(y_values):
                y_values = np.full_like(x_values, y_values)
            
            if not np.all(np.isfinite(y_values)) or np.max(np.abs(y_values)) > FILTER_Y_MAX_ABS_VALUE:
                continue

            data_points = np.column_stack((x_values, y_values))
            valid_data.append({'root': root, 'data': data_points})
        except (TypeError, ValueError, OverflowError):
            continue

    # 3. Filter for Duplicates
    final_dataset = []
    unique_tree_strs = set()
    for item in tqdm(valid_data, desc="Filtering duplicates"):
        tree_str = str_tree(item['root'])
        if tree_str not in unique_tree_strs:
            unique_tree_strs.add(tree_str)
            final_dataset.append(item)

    print(f"Generated {len(final_dataset)} unique and valid expressions.")
    return final_dataset