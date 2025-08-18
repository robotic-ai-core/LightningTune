#!/usr/bin/env python
"""
Test runner script for Lightning BOHB.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py --unit       # Run unit tests only
    python run_tests.py --integration # Run integration tests only
    python run_tests.py --e2e        # Run end-to-end tests only
    python run_tests.py --coverage   # Run with coverage report
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(test_type=None, coverage=False, verbose=True):
    """Run tests with pytest."""
    cmd = ["pytest"]
    
    # Add test directory
    if test_type == "unit":
        cmd.append("tests/unit")
    elif test_type == "integration":
        cmd.append("tests/integration")
    elif test_type == "e2e":
        cmd.append("tests/e2e")
    else:
        cmd.append("tests")
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    
    # Add coverage
    if coverage:
        cmd.extend(["--cov=lightning_bohb", "--cov-report=term-missing", "--cov-report=html"])
    
    # Add color
    cmd.append("--color=yes")
    
    # Run tests
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run Lightning BOHB tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--e2e", action="store_true", help="Run end-to-end tests only")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage report")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    
    args = parser.parse_args()
    
    # Determine test type
    test_type = None
    if args.unit:
        test_type = "unit"
    elif args.integration:
        test_type = "integration"
    elif args.e2e:
        test_type = "e2e"
    
    # Run tests
    exit_code = run_tests(
        test_type=test_type,
        coverage=args.coverage,
        verbose=not args.quiet
    )
    
    # Print coverage report location
    if args.coverage and exit_code == 0:
        print("\n" + "="*50)
        print("Coverage report generated:")
        print(f"  HTML: {Path('htmlcov/index.html').absolute()}")
        print("="*50)
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()