#!/usr/bin/env python3
"""
...

"""

__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "AGPL-3.0"


from ..TextClassifier import TextClassifier
from ..Model import BinaryClassKeyWordModel
from ..Coordinator import SimplePassThruCoord, FirstHitCoord
from ..utils import load_txt

from pathlib import Path


filepath = Path('data/test.txt')
record_presentation_doc = {'clean_body': load_txt(filepath)}
config = {
    'TRAINING_DATA_DIR':{
        'model_topic': {
            'template1': Path('./models_data/template1/pos_kw.txt'),
            'template2': Path('./models_data/template2/pos_kw.txt'),
            'display': '...'
        }
    }
}


def test_text_classifier_single_model():
    model = 'template1'
    model1 = BinaryClassKeyWordModel(model, config['TRAINING_DATA_DIR']['model_topic'][model])
    coord = SimplePassThruCoord()
    tc = TextClassifier(name='model_topic', config=config, models=[model1], coordinator=coord)
    txt = record_presentation_doc['clean_body']
    results = tc.run(txt)
    assert len(results) == 6
    item = results[0]
    del item['timestamp']
    del item['pred']
    del item['target']
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
    assert len(results) == 4
    item = results[0]
    del item['timestamp']
    del item['pred']
    del item['target']
    assert item == {
        'search': 'KW',
        'model_topic': 'template1',
        'topic_class': 'pos'
    }