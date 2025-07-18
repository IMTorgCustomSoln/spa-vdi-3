#!/usr/bin/env python3
"""
Test Utility functions

"""

__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "AGPL-3.0"

from ..utils import (
    load_txt,
    prepare_labels,
    score_model_results
)
import copy


def test_load_txt():
    assert True == True

label_type = 'true'
items = [ {'begin':10, 'end': 20} ]
sentences = [
    {'id': 'outside begin,end', 'begin': 0, 'end': 5, 'text': None, f'{label_type}_type':None, f'{label_type}_BIO': []},
    {'id': 'within begin,end', 'begin': 0, 'end': 5, 'text': None, f'{label_type}_type':None, f'{label_type}_BIO': []},
    {'id': 'within begin, outside end', 'begin': 0, 'end': 5, 'text': None, f'{label_type}_type':None, f'{label_type}_BIO': []},
    {'id': 'outside begin, within end', 'begin': 0, 'end': 5, 'text': None, f'{label_type}_type':None, f'{label_type}_BIO': []},
]

def test_prepare_labels():
    results = [
        None,
        None,
        None,
        None
    ]
    for idx, sentence in enumerate(sentences):
        result = prepare_labels(sentence, items, label_type)
        assert result == result[idx]

def test_score_model_results():
    assert True == True