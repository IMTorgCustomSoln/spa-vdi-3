#!/usr/bin/env python3
"""
TaskExport classes

"""

__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "AGPL-3.0"


from .Document import Document

import fitz

from pathlib import Path
import copy

def load_txt(filepath):
    txt = None
    filepath = Path(__file__).parent / filepath
    if filepath.is_file():
        with open(filepath, 'r') as f:
            txt = f.read()
    return txt