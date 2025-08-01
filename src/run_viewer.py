#!/usr/bin/env python3
"""
Simple script to run the line viewer GUI
"""
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from line_viewer import main

if __name__ == "__main__":
    main()
