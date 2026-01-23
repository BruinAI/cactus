
import json
import fileinput
import sys
import re
from collections import defaultdict
import os

def parse_profile(file_path):
    ops = defaultdict(lambda: {"count": 0, "total_ms": 0.0})
    total_time = 0.0
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found")
        return

    with open(file_path, 'r') as f:
        # The file format from graph_execute.cpp is somewhat custom:
        # === Graph Execution Profile ===
        # Operation       Time (ms)   Output Shape        Backend
        # ------------------------------------------------------------
        # INPUT           0.000       [1, 16000]          values=[...]
        # ...
        
        lines = f.readlines()
        
    start_parsing = False
    
    # Regex to parse the fixed-width-like columns
    # Example line:
    # CONV1D_K7S3     0.123       [1, 64, 112]        values=[...]
    
    for line in lines:
        line = line.strip()
        if "=== Graph Execution Profile ===" in line:
            start_parsing = True
            continue
        if not start_parsing:
            continue
        if line.startswith("---") or line.startswith("Operation") or line.startswith("Total"):
            continue
        if line.startswith("="):
            break
            
        # Split by first few spaces
        parts = line.split(maxsplit=4) # Op, Time, Shape, Rest
        
        if len(parts) >= 2:
            op_name = parts[0]
            try:
                time_ms = float(parts[1])
                ops[op_name]["count"] += 1
                ops[op_name]["total_ms"] += time_ms
            except ValueError:
                continue

    # Print Summary
    print(f"{'Operation':<25} {'Count':<8} {'Total (ms)':<12} {'Avg (ms)':<10} {'% Time':<8}")
    print("-" * 65)
    
    grand_total_ms = sum(d["total_ms"] for d in ops.values())
    
    sorted_ops = sorted(ops.items(), key=lambda x: x[1]["total_ms"], reverse=True)
    
    for op_name, stats in sorted_ops:
        count = stats["count"]
        total = stats["total_ms"]
        avg = total / count if count > 0 else 0
        percent = (total / grand_total_ms * 100) if grand_total_ms > 0 else 0
        
        print(f"{op_name:<25} {count:<8} {total:<12.3f} {avg:<10.3f} {percent:<8.1f}")
        
    print("-" * 65)
    print(f"{'Total':<25} {sum(d['count'] for d in ops.values()):<8} {grand_total_ms:<12.3f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_profile.py <profile_log>")
    else:
        parse_profile(sys.argv[1])
