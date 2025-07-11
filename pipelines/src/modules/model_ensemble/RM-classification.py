#!/usr/bin/env python3
"""
Module Docstring

"""

#TODO:from nltk.tokenize import word_tokenize 

import torch
#from transformers import AutoModel
from setfit import SetFitModel

from pathlib import Path
import copy


#load models
#config_env.config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = Path("BAAI/bge-small-en-v1.5")
model = SetFitModel.from_pretrained(model_path)
model.to(device)



#TODO:add logger
class Classifier:
    """..."""
    def __init__(self):
        pass
    
    def set_config(self, config, model_topics):
        self.config = copy.deepcopy(config)
        self.model_topics = model_topics
        self.classifier = [
            kw_classifier,
            #phrase_classifier,
            #fs_classifier
        ]
        kw_lines = []
        for model_topic, data_dir in self.config['TRAINING_DATA_DIR'].items():
            if model_topic in self.model_topics:
                binary_classifier_tng_data = Path(data_dir / 'pos_kw.txt')
                multi_classifier_tng_data = [item for item in Path(data_dir).rglob('**/*') if item.is_file() and '_sentence' not in str(item)]
                #binary
                if binary_classifier_tng_data.is_file():
                    with open( binary_classifier_tng_data, 'r') as file:
                        kw_lines.extend( file.readlines() )
                    self.config[f'KEYWORDS-{model_topic}-pos'] = [ ' ' + word.replace('\n','') + ' ' for word in kw_lines]      #ensure spacing around word
                    kw_lines = []
                #multi-class (multiple binary)
                elif len(multi_classifier_tng_data) > 2:
                    for file_model_class in multi_classifier_tng_data:
                        with open( file_model_class, 'r') as file:
                            kw_lines.extend( file.readlines() )
                        self.config[f'KEYWORDS-{model_topic}-{str(file_model_class.name)}'] = [ ' ' + word.replace('\n','') + ' ' for word in kw_lines]      #ensure spacing around word
                        kw_lines = []
        
    def run(self, chunk):
        """Importable function to run assigned models."""
        combined_results = []
        for model_topic in self.model_topics:
            for classifier in self.classifier:
                model_topic_result = classifier(self.config, model_topic, chunk)
                if len(model_topic_result) > 0:
                    combined_results.extend(model_topic_result)
        return combined_results
    

TextClassifier = Classifier()



def kw_classifier(config, model_topic, chunk):
    """Apply key word classifier to chunk."""
    default_result = {
        'search': 'KW',
        'topic': model_topic,
        'topic_class': None,
        'target': None,
        'timestamp': None,
        'pred': None
        }
    model_topic_classes = [key for key in config.keys() if model_topic in key]
    results = []
    for topic_class in model_topic_classes:
        result = copy.deepcopy(default_result)
        hits = []
        for word in config[topic_class]:
            word = word.strip()
            if word in chunk['text'].lower():
                hits.append(word)
        #words = word_tokenize(chunk['text'])
        if len(hits)>0:
            result['topic_class'] = topic_class.split('-')[2]
            result['target'] = hits[0]       #TODO: provide formatted chunk['text'], previously: `' '.join(hits)`
            result['pred'] = len(hits) / len(chunk['text'])
            if 'timestamp' in chunk.keys():
                result['timestamp'] = chunk['timestamp']
            results.append(result)
    return results


def phrase_classifier(config, model_topic, chunk):
    """Apply phrase classifiers to chunk."""
    return None


def fs_classifier(config, model_topic, chunk):
    """Apply fs classifier to chunk."""
    result = {
        'search': 'FS',
        'target': None,
        'timestamp': None,
        'pred': None
        }
    if len(chunk['text']) > 40:
        probs = model.predict_proba(chunk['text'])
        pos_idx = model.labels.index('positive')
        prob_positive = probs.tolist()[pos_idx]
        if prob_positive > .5:
            result['target'] = chunk['text']
            result['pred'] = prob_positive
            if 'timestamp' in chunk.keys():
                result['timestamp'] = chunk['timestamp']
            return result
    return None