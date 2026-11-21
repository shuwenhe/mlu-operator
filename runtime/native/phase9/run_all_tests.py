#!/usr/bin/env python3
"""
Phase 9: Validation master script

Runs all unit tests, validates against checkpoint reference,
and generates consolidated acceptance report.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run command and report results."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"FAILED: {description}")
        return False
    print(f"PASSED: {description}")
    return True

def main():
    workspace = Path("/app/shuwen/mlu-operator")
    os.chdir(workspace)
    
    all_passed = True
    
    # Build tests
    build_commands = [
        (
            "g++ -std=c++17 -O2 -fPIC runtime/native/phase9/qk_test.cpp -o artifacts/build/phase9_qk_test",
            "Build QK unit test"
        ),
        (
            "g++ -std=c++17 -O2 -fPIC runtime/native/phase9/softmax_test.cpp -o artifacts/build/phase9_softmax_test",
            "Build Softmax unit test"
        ),
        (
            "g++ -std=c++17 -O2 -fPIC runtime/native/phase9/av_test.cpp -o artifacts/build/phase9_av_test",
            "Build AV unit test"
        ),
    ]
    
    for cmd, desc in build_commands:
        if not run_command(cmd, desc):
            all_passed = False
    
    # Run QK tests
    print(f"\n{'='*60}")
    print("Phase 9.1: QK Tests")
    print(f"{'='*60}")
    
    qk_csv = "artifacts/logs/phase9_qk_results.csv"
    with open(qk_csv, "w") as f:
        f.write("stage,elements,max_error,mean_error,mismatches\n")
    
    for dtype in ["bf16", "fp16"]:
        for seq in [1, 7, 32, 128]:
            cmd = f"./artifacts/build/phase9_qk_test {dtype} {seq} {seq} >> {qk_csv}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  {dtype} seq={seq}: FAIL")
                all_passed = False
            else:
                print(f"  {dtype} seq={seq}: PASS")
    
    # Run Softmax tests
    print(f"\n{'='*60}")
    print("Phase 9.2: Softmax Tests")
    print(f"{'='*60}")
    
    softmax_csv = "artifacts/logs/phase9_softmax_results.csv"
    with open(softmax_csv, "w") as f:
        f.write("stage,elements,max_error,mean_error,mismatches\n")
    
    for dtype in ["bf16", "fp16"]:
        for seq in [1, 7, 32, 128]:
            cmd = f"./artifacts/build/phase9_softmax_test {dtype} {seq} {seq} >> {softmax_csv}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  {dtype} seq={seq}: FAIL")
                all_passed = False
            else:
                print(f"  {dtype} seq={seq}: PASS")
    
    # Run AV tests
    print(f"\n{'='*60}")
    print("Phase 9.3: AV Tests")
    print(f"{'='*60}")
    
    av_csv = "artifacts/logs/phase9_av_results.csv"
    with open(av_csv, "w") as f:
        f.write("stage,elements,max_error,mean_error,mismatches\n")
    
    for dtype in ["bf16", "fp16"]:
        for seq in [1, 7, 32, 128]:
            cmd = f"./artifacts/build/phase9_av_test {dtype} {seq} {seq} >> {av_csv}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  {dtype} seq={seq}: FAIL")
                all_passed = False
            else:
                print(f"  {dtype} seq={seq}: PASS")
    
    # Validate results
    print(f"\n{'='*60}")
    print("Validation Summary")
    print(f"{'='*60}")
    
    validation_commands = [
        (f"python3 runtime/native/phase9/validate_qk.py {qk_csv}", "QK validation"),
        (f"python3 runtime/native/phase9/validate_softmax.py {softmax_csv}", "Softmax validation"),
        (f"python3 runtime/native/phase9/validate_av.py {av_csv}", "AV validation"),
    ]
    
    for cmd, desc in validation_commands:
        if not run_command(cmd, desc):
            all_passed = False
    
    # Final status
    print(f"\n{'='*60}")
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("Phase 9 unit test infrastructure ready for BANG C kernel implementation")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
