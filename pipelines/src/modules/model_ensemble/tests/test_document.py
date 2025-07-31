#!/usr/bin/env python3
"""
Test Utility functions

"""

__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "AGPL-3.0"


from ..Document import (
    Document,
    DocumentFactory
)

from sentence_transformers import SentenceTransformer
from ..TextClassifier import TextClassifier
import fitz

from pathlib import Path
import pytest


filename = Path(__file__).parent.parent / 'data/pdf_open_parameters_acro8.pdf'
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")



def test_doc_sentence():
    doc = DocumentFactory(filename, model, 1)
    assert doc.get_sentences(page=0).__len__() == 4