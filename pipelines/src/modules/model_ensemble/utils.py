#!/usr/bin/env python3
"""
Utility functions for model_ensemble

"""

__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "AGPL-3.0"


from .Document import Document

import fitz

from pathlib import Path
import copy


def load_txt(filepath):
    txt = None
    filepath = Path(__file__).parent / filepath
    if filepath.is_file():
        with open(filepath, 'r') as f:
            txt = f.read()
    return txt


def prepare_labels(sentence, labeled_data, label_type='true'):
    """Prepare true_*/pred_* sentences based on provided labels.
    
    This requires the true_/pred_ dict to list[dict] to include the following keys:
    * 'label'
    * 'begin'
    * 'end'
    * 'text
    """
    items_before_or_after_sent = [item for item in labeled_data if 
        (item['begin'] < sentence['begin'] and
         item['end'] < sentence['begin']) or
         (item['begin'] > sentence['end'] and
         item['end'] > sentence['end'])
         ]
    items_within_sent = [item for item in labeled_data if 
         item['begin'] >= sentence['begin'] and
         item['end'] <= sentence['end']
         ]
    items_begin_before_sent_within_end = [item for item in labeled_data if 
         item['begin'] < sentence['begin'] and
         item['end'] > sentence['begin'] and
         item['end'] < sentence['end']
         ]
    items_begin_within_sent_after_end = [item for item in labeled_data if 
         item['begin'] > sentence['begin'] and
         item['begin'] < sentence['end'] and
         item['end'] > sentence['end']
         ]

    # case: outside begin,end
    if len(items_within_sent)==0  and len(items_begin_before_sent_within_end)==0 and len(items_begin_within_sent_after_end)==0:
        sentence[f'{label_type}_type'] = False
        sentence[f'{label_type}_BIO'] = ['O' for _ in sentence[f'{label_type}_BIO'] if _ == None]
    # case: within begin,end
    elif len(items_within_sent)>0  and len(items_begin_before_sent_within_end)==0 and len(items_begin_within_sent_after_end)==0:
        selected_label = max(items_within_sent, key=lambda item: len(item['text']) )
        sentence[f'{label_type}_type'] = selected_label['label']
        begin = sentence['begin'] - selected_label['begin']
        end = sentence['end'] - selected_label['end']
        body = end - begin
        head_BIO = ['O' for _ in range(begin) if _ == None]
        body_BIO = [f'I-{selected_label["label"]}' for _ in range(body)]
        body_BIO[0] = f'B-{selected_label["label"]}'
        tail_BIO = ['O' for _ in range(end) if _ == None]
        sentence[f'{label_type}_BIO'] = head_BIO + body_BIO + tail_BIO
    # case: within begin, outside end
    elif len(items_within_sent)==0  and len(items_begin_before_sent_within_end)>0 and len(items_begin_within_sent_after_end)==0:
        selected_label = max(items_begin_before_sent_within_end, key=lambda item: len(item['text']) )
        sentence[f'{label_type}_type'] = selected_label['label']
        begin = sentence['begin'] - selected_label['begin']
        end = sentence['end'] - sentence['begin']       #item end is truncated by sentence
        body = end - begin
        head_BIO = ['O' for _ in range(begin) if _ == None]
        body_BIO = [f'I-{selected_label["label"]}' for _ in range(body)]
        body_BIO[0] = f'B-{selected_label["label"]}'
        tail_BIO = ['O' for _ in range(end) if _ == None]
        sentence[f'{label_type}_BIO'] = head_BIO + body_BIO + tail_BIO
    # case: outside begin, within end
    elif len(items_within_sent)==0  and len(items_begin_before_sent_within_end)==0 and len(items_begin_within_sent_after_end)>0:
        selected_label = max(items_begin_within_sent_after_end, key=lambda item: len(item['text']) )
        sentence[f'{label_type}_type'] = selected_label['label']
        begin = 0       #item begin is truncated by sentence
        end = selected_label['end'] - sentence['end']
        body = end - begin
        head_BIO = ['O' for _ in range(begin) if _ == None]
        body_BIO = [f'I-{selected_label["label"]}' for _ in range(body)]
        body_BIO[0] = f'B-{selected_label["label"]}'
        tail_BIO = ['O' for _ in range(end) if _ == None]
        sentence[f'{label_type}_BIO'] = head_BIO + body_BIO + tail_BIO
    return sentence


import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from nervaluate.evaluate import Evaluator

def score_model_results(labeled_data, pred_results, doc):
    """..."""
    results = {}
    #calibrate text
    checks = {}
    for item in labeled_data:
        target_page = item['page'] - 1
        if target_page not in checks.keys():
            checks[target_page] = []
        doc_page_sentences = doc.get_sentences(page=target_page)
        doc_initial_text = doc_page_sentences[0][:100]
        doc_page_chars = ''.join(doc_page_sentences)
        check1 = doc_initial_text in item['calibrateSubString'].lower()
        check2 = item['calibratePageLength'] >= len(doc_page_chars)
        checks[target_page].extend([check1, check2])
    flattened_checks = [item for sublist in checks.values() for item in sublist]
    assert all( flattened_checks ) == True

    #generate standard 'sentence' unit
    target_pages = list(checks.keys())
    true_sentences = []
    pred_sentences = []
    for page in target_pages:
        page_sentences = doc.get_sentences(page=page)
        for idx, sentence in enumerate(page_sentences):
            #prepare sentences
            label_type = 'true'
            true_sent = {}
            true_sent['text'] = copy.deepcopy(sentence)
            tokenized_text = word_tokenize(true_sent['text'])
            true_sent['begin'] = 0 if idx==0 else true_sentences[idx-1]['end'] + 1
            true_sent['end'] = true_sent['begin'] + len(true_sent['text'])
            true_sent[f'{label_type}_type'] = None
            true_sent[f'{label_type}_BIO'] = [None for _ in range(len(tokenized_text))]
            #add correct labels
            true_sent = prepare_labels(true_sent, labeled_data, label_type='true')
            true_sentences.append(true_sent)
            #prepare sentences
            label_type = 'pred'
            pred_sent = {}
            pred_sent['text'] = copy.deepcopy(sentence)
            tokenized_text = word_tokenize(pred_sent['text'])
            pred_sent['begin'] = 0 if idx==0 else pred_sentences[idx-1]['end'] + 1
            pred_sent['end'] = pred_sent['begin'] + len(pred_sent['text'])
            pred_sent[f'{label_type}_type'] = None
            pred_sent[f'{label_type}_BIO'] = [None for _ in range(len(tokenized_text))]
            #add correct labels
            if idx==32:
                print('hi')
            pred_sent = prepare_labels(pred_sent, pred_results, label_type='pred')
            pred_sentences.append(pred_sent)

    #ner IOB matrix scores
    labels = ['template1','template2']
    trues = [true_sent['true_BIO'] for true_sent in true_sentences]
    preds = [pred_sent['pred_BIO'] for pred_sent in pred_sentences]
    evaluator = Evaluator(trues, preds, tags=labels, loader='list')
    results, results_per_tag, result_indices, result_indices_by_tag = evaluator.evaluate()
    results['IOB_matrix'] = evaluator

    #label confusion matrix scores
    scores = {'TP':0, 'FP':0,  'TN':0, 'FN':0}
    results['label_matrix'] = scores

    return results