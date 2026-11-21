#!/usr/bin/env python3
"""
Phase 9.1: QK Validation Script

Verifies all QK test cases pass acceptance criteria.
"""

import sys
import pandas as pd

def main():
    if len(sys.argv) != 2:
        print("Usage: validate_qk.py <results.csv>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Failed to read {csv_file}: {e}")
        sys.exit(1)
    
    expected_stages = ['qk_fp32', 'qk_scaled', 'qk_masked']
    all_passed = True
    
    print("Phase 9.1: QK Acceptance Validation")
    print("=" * 60)
    
    for stage in expected_stages:
        subset = df[df['stage'] == stage]
        
        if subset.empty:
            print(f"✗ {stage}: MISSING (no test cases)")
            all_passed = False
            continue
        
        total_cases = len(subset)
        mismatches_col = subset['mismatches']
        total_mismatches = mismatches_col.sum()
        max_mismatches = mismatches_col.max()
        
        if total_mismatches == 0:
            print(f"✓ {stage}: PASS ({total_cases} cases, 0 mismatches)")
        else:
            print(f"✗ {stage}: FAIL ({total_cases} cases, {total_mismatches} total mismatches, max {max_mismatches})")
            # Print failing cases
            failing = subset[subset['mismatches'] > 0]
            for idx, row in failing.iterrows():
                print(f"    - {row['stage']}: {row['mismatches']} mismatches (max_error={row['max_error']:.9g})")
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("✓ PHASE 9.1 QK: ACCEPTANCE PASS")
        return 0
    else:
        print("✗ PHASE 9.1 QK: ACCEPTANCE FAIL")
        return 1

if __name__ == "__main__":
    sys.exit(main())
