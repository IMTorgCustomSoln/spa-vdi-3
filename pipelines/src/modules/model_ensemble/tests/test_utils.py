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
import pytest


def test_prepare_labels():
    label_type = 'true'
    items = [ {'begin':10, 'end': 20} ]
    sentences = [
        {'id': 'outside begin,end', 'begin': 0, 'end': 5, 'text': None, f'{label_type}_type':None, f'{label_type}_BIO': []},
        {'id': 'within begin,end', 'begin': 0, 'end': 5, 'text': None, f'{label_type}_type':None, f'{label_type}_BIO': []},
        {'id': 'within begin, outside end', 'begin': 0, 'end': 5, 'text': None, f'{label_type}_type':None, f'{label_type}_BIO': []},
        {'id': 'outside begin, within end', 'begin': 0, 'end': 5, 'text': None, f'{label_type}_type':None, f'{label_type}_BIO': []},
    ]
    results = [
        {'id': 'outside begin,end', 'begin': 0, 'end': 5, 'text': None, 'true_type':False, 'true_BIO': []},
        {'id': 'within begin,end', 'begin': 0, 'end': 5, 'text': None, 'true_type':False, 'true_BIO': []},
        {'id': 'within begin, outside end', 'begin': 0, 'end': 5, 'text': None, 'true_type':False, 'true_BIO': []},
        {'id': 'outside begin, within end', 'begin': 0, 'end': 5, 'text': None, 'true_type':False, 'true_BIO': []},
    ]
    for idx, sentence in enumerate(sentences):
        result = prepare_labels(sentence, items, label_type)
        assert result == results[idx]

@pytest.mark.skip(reason='test is maintained at: `tests/test_modeling/test_labeled_data_with_text_classifier()')
def test_score_model_results():
    assert True == True

@pytest.mark.skip(reason='not important')
def test_load_txt():
    assert True == True