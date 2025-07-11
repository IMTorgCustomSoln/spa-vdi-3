#!/usr/bin/env python3
"""
Model Template (abstract) for future models to comply for running with TextClassifier.


Prepare all models used in workflow

Perform the following tasks:
* validate exact-term search items
* finetune models
* obtain results applyig system to test dataset
* log progress and results
"""

__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "AGPL-3.0"

from src.Files import File

import torch
from setfit import SetFitModel

from pathlib import Path
import copy


from config._constants import (
    logger
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = Path("BAAI/bge-small-en-v1.5")
model = SetFitModel.from_pretrained(model_path)
model.to(device)


class Model:
    """..."""

    _default_result = {
        'search': None,
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
            staged_result['model_topic'] = self.model_topic
            self.staged_result = staged_result
        else:
            staged_result = copy.deepcopy(self.staged_result)
        return staged_result
    
    def _validate_key_terms(self, config):
        """..."""
        wdir = config['TRAINING_DATA_DIR'][self.model_topic]
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
        wdir = config['TRAINING_DATA_DIR'][self.model_topic]
        if not wdir.is_dir():
            logger.info(f'model data working dir is not available: {wdir}')
            return False
        #get records for train / test 
        from datasets import load_dataset, Dataset
        file_paths = [wdir / item.name for item in wdir.iterdir() if
                      item.is_file() and
                      ('sentences' in item.name or item.suffix == '.json')
                      ]
        labels = [item.name.split('_sentences.txt')[0] for item in file_paths if
                  'sentences' in item.name
                  ]
        training_files = [item for item in file_paths if
                          'sentences' in item.name
                          ]
        if len(file_paths)==0 or len(labels)==0 or len(training_files)==0:
            return False
        else:
            return True
        
    def _finetune_classification_model(config, model_topic):
        """..."""
        #config_env.config()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'finetune() using device: {device}')
        wdir = config['TRAINING_DATA_DIR'][model_topic]
        if not wdir.is_dir():
            logger.info(f'model data working dir is not available: {wdir}')
            return False

        #get records for train / test 
        from datasets import load_dataset, Dataset
        file_paths = [wdir / item.name for item in wdir.iterdir() if
                      item.is_file() and
                      ('sentences' in item.name or item.suffix == '.json')
                      ]
        labels = [item.name.split('_sentences.txt')[0] for item in file_paths if
                  'sentences' in item.name
                  ]
        training_files = [item for item in file_paths if
                          'sentences' in item.name
                          ]

        #load model
        from setfit import SetFitModel
        load_foundation_model_path = "BAAI/bge-small-en-v1.5"
        save_finetuned_model_path = f"pretrained_models/finetuned--BAAI-{model_topic}"
        if Path(save_finetuned_model_path).is_dir():
            logger.info(f'model is cached and previously refined: {save_finetuned_model_path}')
            return True
        if len(labels)<=2:
            model = SetFitModel.from_pretrained(load_foundation_model_path)
        else:
            model = SetFitModel.from_pretrained(load_foundation_model_path, multi_target_strategy="one-vs-rest")
        model.to(device)
        model.labels = labels
        '''
        if path_tng_records.is_file():
            with open(path_tng_records, 'r') as file:
                train_records = json.load(file)['records']
            #train_dataset = load_dataset(records)         #<<<FAILS HERE, maybe use this: Dataset.from_dict(
        else:
            logger.info(f'no training records available to refine model: {path_tng_records}')
            return False

        if path_test_records.is_file():
            with open(path_test_records, 'r') as file:
                test_records = json.load(file)['records']
            test_dataset = Dataset.from_list(test_records)
        '''
        train_records = []
        for tng_file in training_files:
            label = tng_file.name.split('_sentences.txt')[0]
            with open(tng_file, 'r') as file:
                train_lines = file.readlines()
            recs = [{'text': line.replace('\n',''), 'label':label} for line in train_lines]
            train_records.extend(recs)
        train_dataset = Dataset.from_list(train_records)

        #train model
        from setfit import Trainer, TrainingArguments
        args = TrainingArguments(
            batch_size=25,
            num_epochs=10,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
        )
        trainer.train()

        #test model
        metrics = trainer.evaluate(test_dataset)
        print(metrics)
        '''
        preds = model.predict([
            "I got the flu and felt very bad.",
            "I got a raise and feel great.",
            "This bank is awful.",
            ])
        print(f'predictions: {preds}')
        '''

        #save model
        model_path = Path(save_finetuned_model_path)
        model.save_pretrained(model_path )
        model2 = SetFitModel.from_pretrained(model_path )
        result = True
        if not model2:
            result = False
        return result
    


from pathlib import Path
import re
import time

class BinaryClassKeyWordModel(Model):
    """Simple binary class model based on key words."""

    def __init__(self, model_topic, filepath_to_kw):
        super().__init__(model_topic)
        kw_lines = []
        self.key_words = set()
        filepath_to_kw = Path() / filepath_to_kw
        if not filepath_to_kw.is_file():
            raise Exception(f'ERROR: {filepath_to_kw} is not a file')
        else:
            with open(filepath_to_kw, 'r') as f:
                kw_lines.extend( f.readlines() )
                self.key_words = set( [word.replace('\n', '') + ' ' for word in kw_lines] )
        self.staged_result['search'] = 'KW'
        self.staged_result['topic_class'] = 'pos'

    def validate(self, config):
        return super()._validate_key_terms(config)
    
    def run(self, text):
        results = []
        for key_word in self.key_words:
            hits = re.findall(key_word, text)
            if len(hits) > 0:
                staged_results = self._get_staged_result()
                results = [{
                    'search': staged_results['search'],
                    'model_topic': staged_results['model_topic'],
                    'topic_class': staged_results['topic_class'],
                    'target': key_word,
                    'timestamp': time.time(),
                    'pred': len(hits) / len(text),
                } for hit in hits]
                results.extend(results)
        return results
    


#TODO: class MultiClassKeyWorkModel(Model):




#%pip install --upgrade --quiet langchain-text-splitters tiktoken
from langchain_text_splitters import CharacterTextSplitter
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=1000, chunk_overlap=100
)

class ClassificationModel(Model):
    """Most simple few-shot, classification model."""
    def __init__(self, model_topic, filepath_to_training_data, model):
        super().__init__(model_topic)
        self.training_data = []
        self.staged_result['search'] = 'CM'
        self.staged_result['topic_class'] = 'pos'

    def validate(self, config):
        return super()._valdate_classification_date(config)
    
    def run(self, text):
        results = []
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            probs = model.predict_proba(chunk)
            pos_idx = model.labels.index('positive')
            prob_positive = probs.tolist()[pos_idx]
            if prob_positive > 0.5:
                result = self.get_staged_result()
                result['target'] = chunk['text']
                result['pred'] = prob_positive
                if 'timestamp' in chunk.keys():
                    result['timestamp'] = chunk['timestamp']
                return result
            return None