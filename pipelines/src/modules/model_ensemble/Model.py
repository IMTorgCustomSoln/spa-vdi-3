#!/usr/bin/env python3
"""
Model Template (abstract) for future models to comply for running with TextClassifier.


Prepare all models used in workflow

Perform the following tasks:
* validate exact-term search items
* finetune models
* obtain results applying system to test dataset
* log progress and results
"""

__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "AGPL-3.0"

from src.Files import File

import torch
from setfit import SetFitModel, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
#%pip install --upgrade --quiet langchain-text-splitters tiktoken
from langchain_text_splitters import CharacterTextSplitter

from pathlib import Path
import copy
import random
import string
import json


#config
from config._constants import (
    logger
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
load_foundation_model_path = [
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2"
]
model_path = load_foundation_model_path[1]
model = SetFitModel.from_pretrained(model_path)
model.to(device)

#supporting logic
alphanumeric = string.ascii_letters + string.digits


class Model:
    """..."""

    _default_result = {
        'search': None,
        'id': None,
        'model_topic': None,
        'topic_class': None,
        'target': None,
        'timestamp': None,
        'pred': None
    }

    def __init__(self, model_topic):
        self.model_topic = model_topic
        self.staged_result = None
        self._get_staged_result()

    def _get_staged_result(self):
        if not self.staged_result:
            staged_result = copy.deepcopy(Model._default_result)
            staged_result['id'] = ''.join(random.choices(population=alphanumeric, k=5))
            staged_result['model_topic'] = self.model_topic
            self.staged_result = staged_result
        else:
            staged_result = copy.deepcopy(self.staged_result)
        return staged_result
    
    def _validate_key_terms(self, config):
        """..."""
        wdir = config['TRAINING_DATA_DIR'][self.model_topic][self.model_topic]
        path_pos_keywords = wdir / 'pos_kw.txt'  
        path_neg_keywords = wdir / 'neg_kw.txt'   
        if path_pos_keywords.is_file():
            pos_file = File(path_pos_keywords, 'txt')
            pos_kw = [line.rstrip() for line in pos_file.load_file(return_content=True)]
            logger.info(f'positive keywords found: {len(pos_kw)}')
        else:
            logger.info(f'no positive keywords found at path: {path_pos_keywords}')
        if path_neg_keywords.is_file():
            neg_file = File(path_neg_keywords, 'txt')
            neg_kw = [line.rstrip() for line in neg_file.load_file(return_content=True)]
            logger.info(f'negative keywords found: {len(neg_kw)}')
        else:
            logger.info(f'no negative keywords found at path: {path_neg_keywords}')
        return True
    
    def _validate_classification_data(self, config):
        """..."""
        #config_env.config()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'finetune() using device: {device}')
        wdir = config['TRAINING_DATA_DIR']['model_topic'][self.model_topic]
        if not wdir.is_dir():
            logger.info(f'model data working dir is not available: {wdir}')
            return False
        #get records for train / test 
        file_paths = [item for item in wdir.iterdir() if
                      item.is_file() and
                      ('sentence' in item.name or item.suffix == '.json')
                      ]
        labels = [item.name.split('_sentence.txt')[0] for item in file_paths if
                  'sentence' in item.name
                  ]
        training_files = [item for item in file_paths if
                          'sentence' in item.name
                          ]
        test_files = [item for item in file_paths if 
                      'test.json' in item.name
                      ]
        if len(file_paths)==0 or len(labels)==0 or len(training_files)==0:
            return False
        else:
            self.file_paths = file_paths
            self.labels = labels
            self.training_files = training_files
            self.test_files = test_files
            return True
    
    def _finetune_classification_model(self, config):
        """..."""
        #prepare data
        check = self._validate_classification_data(config)
        load_foundation_model_path = "sentence-transformers/all-MiniLM-L6-v2"
        model_name = load_foundation_model_path.split('/')[1]
        save_finetuned_model_path = Path( f"pretrained_models/finetuned__{model_name}__{self.model_topic}" )
        #training records (required)
        train_records = []
        for training_file in self.training_files:
            label = training_file.name.split('_sentence.txt')[0]
            if training_file.is_file():
                with open(training_file, 'r') as file:
                    train_lines = file.readlines()
                recs = [
                    {'text': line.replace('\n',''), 'label':label} 
                    for line in train_lines
                    ]
                train_records.extend(recs)
            else:
                raise Exception('ERROR: no training files exists')
        train_dataset = Dataset.from_list(train_records)
        #test records (required)
        test_records = []
        for test_file in self.test_files:
            if test_file.is_file():
                with open(test_file, 'r') as file:
                    recs = json.load(file)['records']
                test_records.extend(recs)
            else:
                raise Exception('ERROR: no `test.json` file exists')
        test_dataset = Dataset.from_list(test_records)
        #prepare model
        if save_finetuned_model_path.is_dir():
            logger.info(f'model is cached and previously refined: {save_finetuned_model_path}')
        else:
            if len(self.labels) <= 2:
                model = SetFitModel.from_pretrained(load_foundation_model_path)
            else:
                model = SetFitModel.from_pretrained(load_foundation_model_path, multi_target_strategy="one-vs-rest")
            model.to(device)
            model.labels = self.labels
            #train model
            args = TrainingArguments(
                batch_size=15,
                num_epochs=2
            )
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_dataset
            )
            trainer.train()
            #test and save model
            metrics = trainer.evaluate(test_dataset)
            logger.info(f'metrics: {metrics}')
            model.save_pretrained(save_finetuned_model_path)
        #test model
        model2 = SetFitModel.from_pretrained(save_finetuned_model_path)
        result = True
        if not model2:
            result = False
        self.model = model2
        return result
    


from pathlib import Path
import re
import time

class BinaryClassKeyWordModel(Model):
    """Simple binary class model based on key words."""

    def __init__(self, model_topic, filepath_to_kw):
        super().__init__(model_topic)
        self._set_keywords(filepath_to_kw)
    
    def _set_keywords(self, filepath_to_kw):
        kw_lines = []
        self.key_words = set()
        filepath_to_kw = Path(filepath_to_kw)
        files = [item for item in filepath_to_kw.iterdir() if 'pos_kw' in item.stem]    #TODO:add neg_kw.txt also
        if len(files) < 1:
            raise Exception(f'ERROR: {filepath_to_kw} does not contain a `_kw` named file')
        else:
            for file in files:
                label = file.stem.split('_kw')[0]
                with open(file, 'r') as f:
                    kw_lines.extend( f.readlines() )
                    self.key_words = set( [word.replace('\n', '') + ' ' for word in kw_lines] )
            self.staged_result['search'] = 'KW'
            self.staged_result['topic_class'] = label

    def validate(self, config):
        return super()._validate_key_terms(config)
    
    def run(self, text):
        results = []
        for key_word in self.key_words:
            hits = [ *re.finditer(key_word, text) ]
            if len(hits) > 0:
                staged_results = self._get_staged_result()
                results = [{
                    'search': staged_results['search'],
                    'model_topic': staged_results['model_topic'],
                    'topic_class': staged_results['topic_class'],

                    'label': staged_results['model_topic'],
                    'begin': hit.start(),
                    'end': hit.end(),
                    'text': hit.group(),

                    'target': key_word,
                    'timestamp': time.time(),
                    'pred': len(hits) / len(text),
                } for hit in hits]
                results.extend(results)
        return results
    


#TODO: class MultiClassKeyWorkModel(Model):





text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=1000, chunk_overlap=100
)

class ClassificationModel(Model):
    """Most simple few-shot, classification model."""
    def __init__(self, model_topic, filepath_to_training_data):
        super().__init__(model_topic)
        self.training_data = []
        self._set_training_data(filepath_to_training_data)

    def _set_training_data(self, filepath_to_training_data):
        kw_lines = []
        self.key_words = set()
        filepath_to_kw = Path() / filepath_to_training_data
        files = [item for item in filepath_to_kw.iterdir() if 'pos_sentence' in item.stem]
        if len(files) < 1:
            raise Exception(f'ERROR: {filepath_to_kw} does ont contain a `_sentence` named file')
        else:
            for file in files:
                label = file.stem.split('_sentence')[0]
                with open(file, 'r') as f:
                    kw_lines.extend( f.readlines() )
                    self.training_data = set( [word.replace('\n', '') + ' ' for word in kw_lines] )
            self.staged_result['search'] = 'CM'
            self.staged_result['topic_class'] = label

    def validate(self, config):
        return super()._validate_classification_data(config)

    def finetune(self, config):
        return super()._finetune_classification_model(config)

    def run(self, text):
        results = []
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            probs = self.model.predict_proba(chunk)
            pos_idx = self.model.labels.index('pos')
            prob_positive = probs.tolist()[pos_idx]
            if prob_positive > 0.5:
                result = self.get_staged_result()
                result['target'] = chunk['text']
                result['pred'] = prob_positive
                if 'timestamp' in chunk.keys():
                    result['timestamp'] = chunk['timestamp']
                results.append( result )
            results.append( None )
        return results