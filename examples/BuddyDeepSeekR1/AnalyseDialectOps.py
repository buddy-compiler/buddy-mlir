#!/usr/bin/env python3

import os
import re
from collections import Counter, defaultdict
from pathlib import Path

def extract_dialect_ops(mlir_file_path):
    """
    Extract operations from all dialects in an MLIR file and count their occurrences.
    
    Args:
        mlir_file_path (str): Path to the MLIR file
        
    Returns:
        dict: Dictionary containing dialect names as keys and Counter objects as values
    """
    # Read the MLIR file
    with open(mlir_file_path, 'r') as f:
        content = f.read()
    
    # Find all operations using regex
    # This pattern matches lines that contain operation names with dialect prefix
    # Excludes numbers and common non-dialect prefixes
    op_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)'
    all_ops = re.findall(op_pattern, content)
    
    # Group operations by dialect
    dialect_ops = defaultdict(Counter)
    for dialect, op in all_ops:
        # Skip common non-dialect prefixes
        if dialect.lower() in ['func', 'module', 'memref', 'arith', 'builtin']:
            continue
        dialect_ops[dialect][op] += 1
    
    return dialect_ops

def main():
    # Get the directory of the current script
    current_dir = Path(__file__).parent
    
    # Construct path to subgraph0.mlir
    mlir_file = current_dir / 'subgraph0.mlir'
    
    if not mlir_file.exists():
        print(f"Error: {mlir_file} not found")
        return
    
    # Extract and count operations by dialect
    dialect_ops = extract_dialect_ops(str(mlir_file))
    
    # Print results
    print("\nMLIR Operation Statistics:")
    print("=" * 60)
    print(f"{'Dialect':<20} {'Operation':<30} {'Count':<10}")
    print("=" * 60)
    
    total_ops = 0
    total_unique_ops = 0
    
    # Sort dialects by total operation count
    sorted_dialects = sorted(
        dialect_ops.items(),
        key=lambda x: sum(x[1].values()),
        reverse=True
    )
    
    for dialect, ops in sorted_dialects:
        dialect_total = sum(ops.values())
        total_ops += dialect_total
        total_unique_ops += len(ops)
        
        print(f"\n{dialect} (Total: {dialect_total} ops)")
        print("-" * 60)
        
        # Sort operations by count
        sorted_ops = sorted(ops.items(), key=lambda x: x[1], reverse=True)
        for op, count in sorted_ops:
            print(f"{'':<20} {op:<30} {count:<10}")
    
    print("\n" + "=" * 60)
    print(f"Total dialects: {len(dialect_ops)}")
    print(f"Total unique operations: {total_unique_ops}")
    print(f"Total operation instances: {total_ops}")

if __name__ == "__main__":
    main()
