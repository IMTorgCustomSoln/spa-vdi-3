"""
Module Docstring


TODO: use sklearn to not repeat logic
* metrics - f1_scores, etc.
* model selection - train_test_split
* base.py - base model for mixins to be added
"""

from src.modules.model_ensemble.utils import (
    load_txt,
    prepare_labels,
    score_model_results
)
from src.io.utils import xform_VDI_NotesData_to_page_labels
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
filename = Path(__file__).parent / 'data/credit-protection-agreement.pdf'
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
doc = DocumentFactory(filename, model, 1)
record_presentation_doc = {'clean_body': ''.join(doc.get_sentences(page=0)) }
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
    #setup model for inference
    model_name1 = 'template1'
    model1 = BinaryClassKeyWordModel(model_name1, config['TRAINING_DATA_DIR']['model_topic'][model_name1])
    model_name2 = 'template2'
    model2 = BinaryClassKeyWordModel(model_name2, config['TRAINING_DATA_DIR']['model_topic'][model_name2])
    coord = FirstHitCoord()
    tc = TextClassifier(name='model_topic', config=config, models=[model1,model2], coordinator=coord)
    #run inference
    txt = record_presentation_doc['clean_body']
    pred_results = tc.run(txt)
    assert 'label' in pred_results[0].keys()
    assert 'begin' in pred_results[0].keys()
    assert 'end' in pred_results[0].keys()
    assert 'text' in pred_results[0].keys()
    #review scores
    results = score_model_results(labeled_data, pred_results, doc)
    assert 'label_matrix' in results.keys()
    assert 'IOB_matrix' in results.keys()
    assert True == True