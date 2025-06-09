#!/usr/bin/env python3

import sys
import os

# Assure l'import depuis la racine
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from environments.multi_stochastic import demo_environment

if __name__ == "__main__":
    demo_environment()
