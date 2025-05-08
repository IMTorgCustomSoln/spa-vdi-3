#!/usr/bin/env python3
"""
Test PipelineRecord for atomic record for Tasks.

"""

__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "AGPL-3.0"

from src.Task import PipelineRecordFactory, PipelineRecord


def test_pipelinerecord_creation():
    id = '12345'
    source_type = 'single_file'
    factory = PipelineRecordFactory()
    record = factory.create_from_id(id, source_type)
    assert type(record) == PipelineRecord