#!/usr/bin/env python3
"""
Test TextClassifier and the Models and Coordinators it uses

"""

__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "AGPL-3.0"


from ..TextClassifier import TextClassifier
from ..Model import BinaryClassKeyWordModel, ClassificationModel
from ..Coordinator import SimplePassThruCoord, FirstHitCoord
from ..utils import load_txt
from ..Document import (
    Document,
    DocumentFactory
)

from pathlib import Path
import pytest


filepath = Path('data/test.txt')
record_presentation_doc = {'clean_body': load_txt(filepath)}
config = {
    'TRAINING_DATA_DIR':{
        'model_topic': {
            'template1': Path('./models_data/template1'),
            'template2': Path('./models_data/template2'),
            'complaints': Path('./models_data/complaints'),
            'display': '...'
        }
    }
}

"""TODO: change model signature to:
model_name = 'model1'
data_topic = 'complaints'
model1 = BinaryClassKeyWordModel(
    model_name,
    data_topic,
    config['TRAINING_DATA_DIR']['model_topic'][model]
    )
"""


def test_text_classifier_single_model():
    model = 'template1'
    model1 = BinaryClassKeyWordModel(model, config['TRAINING_DATA_DIR']['model_topic'][model])
    coord = SimplePassThruCoord()
    tc = TextClassifier(name='model_topic', config=config, models=[model1], coordinator=coord)
    txt = record_presentation_doc['clean_body']
    results = tc.run(txt)
    assert len(results) >= 4
    item = results[0]
    for key in ['timestamp', 'pred', 'target', 'label', 'begin', 'end', 'text']: del item[key]
    assert item == {
        'search': 'KW',
        'model_topic': 'template1',
        'topic_class': 'pos'
    }

def test_text_classifier_multiple_models():
    model_name1 = 'template1'
    model1 = BinaryClassKeyWordModel(model_name1, config['TRAINING_DATA_DIR']['model_topic'][model_name1])
    model_name2 = 'template2'
    model2 = BinaryClassKeyWordModel(model_name2, config['TRAINING_DATA_DIR']['model_topic'][model_name2])
    coord = FirstHitCoord()
    tc = TextClassifier(name='model_topic', config=config, models=[model1,model2], coordinator=coord)
    txt = record_presentation_doc['clean_body']
    results = tc.run(txt)
    assert len(results) >= 4
    item = results[0]
    for key in ['timestamp', 'pred', 'target', 'label', 'begin', 'end', 'text']: del item[key]
    assert item == {
        'search': 'KW',
        'model_topic': 'template1',
        'topic_class': 'pos'
    }

def test_text_classifier_validate():
    model_name1 = 'template1'
    model1 = BinaryClassKeyWordModel(model_name1, config['TRAINING_DATA_DIR']['model_topic'][model_name1])
    model_name2 = 'template2'
    model2 = ClassificationModel(model_name2, config['TRAINING_DATA_DIR']['model_topic'][model_name2])
    coord = FirstHitCoord()
    tc = TextClassifier(name='model_topic', config=config, models=[model1,model2], coordinator=coord)
    checks = tc.validate_models_input()
    assert all(checks.values()) == True

@pytest.mark.skip(reason='finetuning is too slow to run')
def test_text_classifier_finetune_on_local_file():
    model_name1 = 'template1'
    model1 = BinaryClassKeyWordModel(model_name1, config['TRAINING_DATA_DIR']['model_topic'][model_name1])
    model_name2 = 'template2'
    model2 = ClassificationModel(model_name2, config['TRAINING_DATA_DIR']['model_topic'][model_name2])
    coord = FirstHitCoord()
    tc = TextClassifier(name='model_topic', config=config, models=[model1,model2], coordinator=coord)
    checks = tc.validate_models_input()
    results = tc.finetune_models()
    assert all(results.values()) == True

@pytest.mark.skip(reason='not complete yet')
def test_text_classifier_finetune_on_document():
    model_name1 = 'template1'
    model1 = BinaryClassKeyWordModel(model_name1, config['TRAINING_DATA_DIR']['model_topic'][model_name1])
    model_name2 = 'template2'
    model2 = ClassificationModel(model_name2, config['TRAINING_DATA_DIR']['model_topic'][model_name2])
    coord = FirstHitCoord()
    tc = TextClassifier(name='model_topic', config=config, models=[model1,model2], coordinator=coord)
    checks = tc.validate_models_input()

    """
    doc = DocumentFactory(filename, model, 1)
    training_data = doc.get_training_data(page=0)
    results = tc.finetune_models(training_data)
    assert all(results.values()) == True
    """