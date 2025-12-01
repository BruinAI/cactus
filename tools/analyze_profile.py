#!/usr/bin/env python3
"""Quick analysis of profile.txt output."""

import sys
import re
from collections import defaultdict

def parse_profile(filename):
    ops = []
    with open(filename, 'r') as f:
        for line in f:
            # Match lines like: CONV2D         16.853      [1,16,320,320]
            match = re.match(r'^(\w+)\s+([\d.]+)\s+\[', line.strip())
            if match:
                op_name = match.group(1)
                time_ms = float(match.group(2))
                ops.append((op_name, time_ms))
    return ops

def analyze(ops):
    # Group by operation type
    op_times = defaultdict(list)
    for op_name, time_ms in ops:
        op_times[op_name].append(time_ms)
    
    print("=" * 60)
    print("OPERATION SUMMARY")
    print("=" * 60)
    print(f"{'Operation':<20} {'Count':>6} {'Total (ms)':>12} {'Avg (ms)':>10} {'Min':>8} {'Max':>8}")
    print("-" * 60)
    
    total_time = 0
    conv2d_time = 0
    
    # Sort by total time descending
    sorted_ops = sorted(op_times.items(), key=lambda x: sum(x[1]), reverse=True)
    
    for op_name, times in sorted_ops:
        count = len(times)
        total = sum(times)
        avg = total / count
        min_t = min(times)
        max_t = max(times)
        total_time += total
        
        if op_name == "CONV2D":
            conv2d_time = total
        
        print(f"{op_name:<20} {count:>6} {total:>12.3f} {avg:>10.3f} {min_t:>8.3f} {max_t:>8.3f}")
    
    print("-" * 60)
    print(f"{'TOTAL':<20} {len(ops):>6} {total_time:>12.3f}")
    print()
    
    print("=" * 60)
    print("CONV2D IMPACT ANALYSIS")
    print("=" * 60)
    print(f"Total execution time:     {total_time:.3f} ms")
    print(f"CONV2D total time:        {conv2d_time:.3f} ms")
    print(f"CONV2D percentage:        {100 * conv2d_time / total_time:.1f}%")
    print(f"Time without CONV2D:      {total_time - conv2d_time:.3f} ms")
    print()

if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else "profile.txt"
    ops = parse_profile(filename)
    if not ops:
        print(f"No operations found in {filename}")
        sys.exit(1)
    analyze(ops)

