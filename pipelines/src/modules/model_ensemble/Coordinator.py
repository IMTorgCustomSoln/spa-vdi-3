#!/usr/bin/env python3
"""
Coordinator Template (abstract) for combining model results into an ensemble within a TextClassifier.

"""

__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "AGPL-3.0"

import concurrent.futures


class Coordinator:
    """..."""

    def __init__(self):
        pass

    def run(self, model_futures):
        pass


class SimplePassThruCoord(Coordinator):
    """Pass through any results provided.
    """
    def run(self, model_futures):
        result = model_futures
        return result
    

class FirstHitCoord(Coordinator):
    """"Take the results of the first model
    to complete.  This will minimize runtime
    for a TextClassifier.
    """
    def run(self, model_futures):
        results = []
        done_and_not_done = concurrent.futures.wait(
            model_futures,
            return_when=concurrent.futures.FIRST_COMPLETED
        )
        for done in done_and_not_done.done:
            results.extend( done.result() )
        return results