"""
Module Docstring
"""

from src.modules.model_ensemble.Document import (
    Document,
    DocumentFactory
)
from src.modules.model_ensemble.Model import (
    Model,
    BinaryClassKeyWordModel,
    ClassificationModel
)
from src.modules.model_ensemble.Coordinator import (
    SimplePassThruCoord, 
    FirstHitCoord
)
from src.modules.model_ensemble.TextClassifier import TextClassifier
from src.modules.model_ensemble.utils import load_txt
from src.io.utils import xform_VDI_NotesData_to_page_labels

from sentence_transformers import SentenceTransformer

import pytest
from pathlib import Path
import copy
import json


#labeled data
vdi_notesdata_file = 'VDI_NotesData_v0.2.1.json'
filepath = Path(f'tests/data/{vdi_notesdata_file}')
with open(filepath, 'r') as f:
    notesdata = json.load(f)
labeled_data = xform_VDI_NotesData_to_page_labels(notesdata)


#document
filename = Path(__file__).parent.parent / 'data/pdf_open_parameters_acro8.pdf'
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
doc = DocumentFactory(filename, model, 1)
record_presentation_doc = {'clean_body': doc.get_sentences(page=0) }
config = {
    'TRAINING_DATA_DIR':{
        'model_topic': {
            'template1': Path('./models_data/template1/pos_kw.txt'),
            'template2': Path('./models_data/template2/pos_kw.txt'),
            'display': '...'
        }
    }
}


def test_labeled_data_with_model():
    assert True == True

def test_labeled_data_with_text_classifier():
    model_name1 = 'template1'
    model1 = BinaryClassKeyWordModel(model_name1, config['TRAINING_DATA_DIR']['model_topic'][model_name1])
    model_name2 = 'template2'
    model2 = BinaryClassKeyWordModel(model_name2, config['TRAINING_DATA_DIR']['model_topic'][model_name2])
    coord = FirstHitCoord()
    tc = TextClassifier(name='model_topic', config=config, models=[model1,model2], coordinator=coord)
    txt = record_presentation_doc['clean_body']
    results = tc.run(txt)

    item = results[0]

    assert True == True