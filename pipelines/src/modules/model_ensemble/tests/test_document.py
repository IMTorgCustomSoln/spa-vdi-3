from ..Document import (
    Document,
    DocumentFactory
    #SearchModelAbstract,
    #SearchHybrid,
    #train_test_data
)
#from ..Document import load_doc

from sentence_transformers import SentenceTransformer
from ..TextClassifier import TextClassifier
import fitz

from pathlib import Path
import pytest


filename = Path(__file__).parent.parent / 'data/pdf_open_parameters_acro8.pdf'
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")



def test_doc_class():
    doc = DocumentFactory(filename, model, 1)
    assert doc.get_sentences(page=0).__len__() == 4

"""
def test_doc_search_abstract():
    abstract = SearchModelAbstract()
    cutoff = 0.0
    doc = load_doc(filename, model, 1)
    scores = doc.get_search_scores(abstract, cutoff)
    assert list(scores.keys()).__len__() == 60
    assert scores['0-0']['text'] == 'filler text'

def test_doc_search_hybrid_load_train_test():
    shybrid = SearchHybrid()
    checks = shybrid.train_test(train_test_data)
    assert all(checks) == True
    assert type(shybrid.classifier) == Classifier


def test_doc_search_hybrid():
    #setup
    shybrid = SearchHybrid()
    checks = shybrid.train_test(train_test_data)
    cutoff = 0.0
    doc = load_doc(filename, model, 1)
    #run
    scores = doc.get_search_scores(shybrid, cutoff)
    assert list(scores.keys()).__len__() == 60
    assert scores['0-0']['text'] == 'filler text'
"""