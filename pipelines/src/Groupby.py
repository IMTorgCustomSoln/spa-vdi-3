#!/usr/bin/env python3
"""
Groupby abstract

"""

__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "AGPL-3.0"


from itertools import groupby


class Groupby:
    """..."""
    def __init__(self):
        pass

    def get_file_group_id(self, file):
        pass

    def run(self, files):
        files_sorted = [file for file in 
                        sorted(files, key=lambda x: self.get_file_group_id(x))
                        ]
        files_grouped = {key: list(group) for key, group in 
                         groupby(files_sorted, key=lambda x: self.get_file_group_id(x))
                         }
        return files_grouped