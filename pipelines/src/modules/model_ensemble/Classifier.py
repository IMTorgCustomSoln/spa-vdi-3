#!/usr/bin/env python3
"""
Classifier template for all other classifiers

"""

__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "AGPL-3.0"


class ClassifierTemplate:
    """Base classifier for all encapsulating and / or
    combining models.
    """
    def __init__(self, name, config):
        self.name = name
        self.config = config

    def train(self):
        """Train the models on a tagged dataset and 
        get results and parameters for each model.
        """
        pass

    def test(self):
        """Test the models on a tagged dataset and 
        obtain table of results.
        """
        pass

    def run(self, text):
        result = text
        return result