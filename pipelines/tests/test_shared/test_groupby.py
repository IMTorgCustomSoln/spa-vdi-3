#!/usr/bin/env python3
"""
Test groupby
"""

__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "AGPL-3.0"


from src.Groupby import Groupby

import pytest


def test_groupby():
    #setup
    files = [
        {'name': 'b_file'},
        {'name': 'c_file'},
        {'name': 'b_file'},
        {'name': 'a_file'},
        {'name': 'a_file'},
        {'name': 'a_file'},
    ]
    #new grouping logic
    def simple_name_grouping(self, file):
        return file['name']
    #run
    Groupby.get_file_group_id = simple_name_grouping
    groupby = Groupby()
    files_grouped = groupby.run(files)
    assert files_grouped == {
        'a_file': [{'name': 'a_file'},{'name': 'a_file'},{'name': 'a_file'},],
        'b_file': [{'name': 'b_file'},{'name': 'b_file'},],
        'c_file': [{'name': 'c_file'},],
    }