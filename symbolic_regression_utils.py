# Core utilities for tree manipulation, SymPy conversion, and tokenization.

import sympy as sp
from config import LEAVES

class TreeNode:
    """A node in a binary tree representing a mathematical expression."""
    def __init__(self, val=None, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def str_tree(root, level=0, prefix="Root "):
    """Returns a pretty string representation of a TreeNode expression tree."""
    if isinstance(root, str):
        return " " * (level * 4) + prefix + root + "\n"
    if root is None:
        return ""
    
    s = " " * (level * 4) + prefix + str(root.val) + "\n"
    s += str_tree(root.left, level + 1, "L-- ")
    s += str_tree(root.right, level + 1, "R-- ")
    return s

def tree_to_sympy(root, binary_operators, unary_operators):
    """Converts a TreeNode tree into a SymPy expression."""
    if not isinstance(root, TreeNode):
        assert root in LEAVES
        return sp.Symbol(root)

    if root.val in binary_operators:
        left_expr = tree_to_sympy(root.left, binary_operators, unary_operators)
        right_expr = tree_to_sympy(root.right, binary_operators, unary_operators)
        return binary_operators[root.val](left_expr, right_expr)
    
    elif root.val in unary_operators:
        left_expr = tree_to_sympy(root.left, binary_operators, unary_operators)
        return unary_operators[root.val](left_expr)
    
    else:
        raise ValueError(f"Unknown operator in tree: {root.val}")

def create_token_maps(binary_operators, unary_operators):
    """Creates forward and backward token maps for the vocabulary."""
    all_symbols = (
        list(binary_operators.keys()) +
        list(unary_operators.keys()) +
        list(LEAVES) +
        [None, "<EOS>", "<PAD>"]
    )
    forward_map = {symbol: i for i, symbol in enumerate(all_symbols)}
    backward_map = {i: symbol for i, symbol in enumerate(all_symbols)}
    return forward_map, backward_map

def tree_to_prefix_sequence(root):
    """Converts a TreeNode tree to a prefix notation sequence."""
    if not isinstance(root, TreeNode):
        return [root]
    
    sequence = [root.val]
    if root.left is not None:
        sequence.extend(tree_to_prefix_sequence(root.left))
    if root.right is not None:
        sequence.extend(tree_to_prefix_sequence(root.right))
    return sequence

def prefix_sequence_to_tree(sequence, binary_operators, unary_operators):
    """Converts a prefix sequence back to a TreeNode tree."""
    iterator = iter(sequence)
    
    def build_tree():
        val = next(iterator)
        
        if val in LEAVES or val is None:
            return 1, val

        node = TreeNode(val=val)
        
        left_count, node.left = build_tree()
        
        if val in binary_operators:
            right_count, node.right = build_tree()
            return 1 + left_count + right_count, node
        
        elif val in unary_operators:
            assert node.right is None
            return 1 + left_count, node
        
        raise ValueError(f"Unknown operator in sequence: {val}")

    count, root = build_tree()
    assert count == len(sequence)
    return root