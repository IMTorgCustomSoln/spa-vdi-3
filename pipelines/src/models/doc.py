

from .prepare_models import (
    validate_key_terms

)

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

from sentence_transformers import SentenceTransformer
from src.classification import TextClassifier
import fitz
import sklearn

from pathlib import Path
import copy


config = {
    'TRAINING_DATA_DIR': {
        'coverage': Path('/workspaces/contract-data/models_data/coverage'),
        },
}
train_test_data = {
    'config': config,
    'topics': ['coverage']
}
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


class SearchModelAbstract:
    """..."""
    def __init__(self):
        pass
    def infer(self, sentence):
        score = 0.0
        text = 'filler text'
        return score, text


class SearchHybrid(SearchModelAbstract):
    """..."""
    def __init__(self):
        self.classifier = None

    def train_test(self, train_test_data):
        checks = []
        for topic in train_test_data['topics']:
            check = validate_key_terms(train_test_data['config'], topic)
            checks.append(check)
        if all(checks) == True:
            check = TextClassifier.config(train_test_data['config'])
            if check:
                self.classifier = TextClassifier
            checks.append(check)
        return checks

    def infer(self, sentence):
        result = self.classifier.run(sentence)
        score = result['pred']     #TODO: TypeError: list indices must be integers or slices, not str
        text = result['target']
        return score, text






class Doc:
    """"..."""
    def __init__(self, model):
        self.model = model
        #config
        self.Sentence = {
            'raw': '',
            'text': '',
            'embedding': [],
        }
        default_page = {-1: []}
        self.pages = default_page
        self.remove_page(-1)
        #checks
        check = self.add_page(-1, ['This is a test text.',  'This is also test text.'])
        if check == True:
            self.remove_page(-1)
        
    def add_page(self, page, text_blocks):
        if page in self.pages.keys():
            return False
        else:
            self.pages[page] = []
            block_sentences = []
            for block in text_blocks:
                sents = sent_tokenize(block[4])
                embeddings = self.model.encode(sents)
                if len(sents) > 0:
                    for idx, sent in enumerate(sents):
                        sentence = copy.deepcopy(self.Sentence)
                        sentence['raw'] = sent
                        sentence['text'] = sent.lower()
                        sentence['embedding'] = embeddings[idx]
                        block_sentences.append(sentence)
            self.pages[page].extend(block_sentences)
            return True
        
    def remove_page(self, page):
        del self.pages[page]

    def get_page_count(self):
        return max(self.pages.keys())
    
    def get_sentences(self, key='text', page=None):
        sentences = []
        if page == None:
            items = [page[key] for page in self.pages]
            sentences.extend(items)
        else:
            items = [sent[key] for sent in self.pages[page]]
            sentences.extend(items)
        return sentences
    
    def get_search_scores(self, search_model, cutoff=0.5):
        scores = {}
        for page, sents in self.pages.items():
            for sent_idx, sentence in enumerate(sents):
                score, text = search_model.infer(sentence)
                if score > cutoff:
                    key = f'{page}-{sent_idx}'
                    scores[key] = {'score': score, 'text': text}
        return scores