#!/usr/bin/env python3
"""
Main runner script for the Financial RAG Evaluation System.
This script calls the evaluation runner in the evaluation package.
"""

import sys
from evaluation.run import main

if __name__ == "__main__":
    sys.exit(main()) 