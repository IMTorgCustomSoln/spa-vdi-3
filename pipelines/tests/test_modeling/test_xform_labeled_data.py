"""
Module Docstring
"""

from src.io.utils import xform_VDI_NotesData_to_page_labels

from pathlib import Path
import copy
import json
import pytest


vdi_notesdata_file = 'VDI_NotesData_v0.2.1.json'
filepath = Path(f'tests/data/{vdi_notesdata_file}')
with open(filepath, 'r') as f:
    notesdata = json.load(f)


def test_xform_VDI_NotesData_to_page_labels():
    labeled_data = xform_VDI_NotesData_to_page_labels(notesdata)
    assert len(labeled_data) == 3
    assert list(labeled_data[0].keys()) == ['id', 'label', 'docname', 'page', 'text']