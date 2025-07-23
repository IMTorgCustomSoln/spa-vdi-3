#!/usr/bin/env python3
"""
TaskExport classes

"""

__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "AGPL-3.0"

from src.io import utils
from src.Files import File
from src.Task import Task
from src.modules.model_ensemble import TextClassifier

import pandas as pd

from pathlib import Path
import time
import copy
import shutil


'''
def split_str_into_chunks(str_item, N):
    """Split string into list of equal length chunks."""
    chunks = None
    if type(str_item) == str:
        if len(str_item) > 0:
            chunks = [{'text': str_item[i:i+N]} for i in range(0, len(str_item), N)]
    return chunks




#TODO:put model tasks into a new file
class ApplyTextModelsTask(Task):
    """Apply text models (keyterms, classification, etc.) to document sentences in most 
    simple scenario.
    """

    def __init__(self, config, input, output, model_topics):
        self.model_topics = model_topics
        super().__init__(config, input, output)

    def apply_models_for_substring_results(self, record):
        N = 500
        doc = copy.deepcopy(record.presentation_doc)
        if 'models' not in doc.keys(): doc['models'] = []
        if doc['clean_body'] == None: return doc
        models = []
        chunks = split_str_into_chunks(doc['clean_body'], N)
        if len(chunks)>0:
            for chunk in chunks:
                results = TextClassifier.run(chunk)
                for result in results:
                    if result != None:
                        models.append(result)
                    else:
                        models.append({})
        else:
            models.append({})
        #TODO:fix time_asr
        doc['models'].extend(models)
        doc['time_asr'] = 0
        doc['time_textmdl'] = time.time() - self.config['START_TIME']
        self.config['LOGGER'].info(f'text-classification processed for file {record.id} - {record.root_source}')
        return doc

    def run(self):
        TextClassifier.set_config(self.config, self.model_topics)
        for file in self.get_next_run_file_from_directory():
            check = file.load_file(return_content=False)
            pipe_rec = file.get_content()
            new_presentation_doc = self.apply_models_for_substring_results(pipe_rec)
            pipe_rec.presentation_doc = new_presentation_doc
            self.pipeline_record_ids.append(pipe_rec.id)
            filepath = self.export_pipeline_record_to_file(pipe_rec)
            if filepath:
                self.config['LOGGER'].info(f'saved intermediate file {pipe_rec.id} - {filepath}')
            else:
                self.config['LOGGER'].info(f'failed to save intermediate file {pipe_rec.id}')
        self.config['LOGGER'].info(f'completed text-classification processing for file {len(self.pipeline_record_ids)}')
        return True



from collections import Counter

class ApplyClassifyDocModels(Task):
    """Apply text models (keyterms, classification, etc.) to classify whole documents in most 
    simple scenario.
    """

    def __init__(self, config, input, output, model_topics):
        self.model_topics = model_topics
        super().__init__(config, input, output)

    def apply_models_for_document_results(self, record):
        chunk_length_chars = 500
        first_n_sentences = 100
        doc = copy.deepcopy(record.presentation_doc)
        if 'models' not in doc.keys(): doc['models'] = []
        model_results = []
        if doc['clean_body'] == None: return doc
        chunks = split_str_into_chunks(doc['clean_body'], chunk_length_chars)[:first_n_sentences]
        for chunk in chunks:
            results = TextClassifier.run(chunk)
            clean_results = [result for result in results if result not in [None, {}]]
            model_results.extend(clean_results)
        topics = list(set( [item['topic'] for item in model_results if 'topic' in item.keys()] ))
        
        cummulative_models = []
        for topic in topics:
            pred_topic_classes = [item['topic_class'] for item in model_results 
                                  if 'pred' in item.keys() and 
                                  topic in item['topic']
                                  ]
            if len(pred_topic_classes)>0:
                pred_counts = Counter(pred_topic_classes )
                item = {
                    'search': f"Doc-{topic}",
                    'target': pred_counts.most_common()[0][0],
                    'timestamp': None,
                    'pred': pred_counts.most_common()[0][1]
                }
            else:
                item = {
                    'search': f'Doc-{topic}',
                    'target': '*** ERROR ***',
                    'timestamp': None,
                    'pred': '*** ERROR ***',
                }
            cummulative_models.append(item)
        #TODO:fix time_asr
        doc['models'].extend(cummulative_models)
        doc['time_asr'] = 0
        doc['time_textmdl'] = time.time() - self.config['START_TIME']
        self.config['LOGGER'].info(f'text-classification processed for file {record.id} - {record.root_source}')
        return doc

    def run(self):
        TextClassifier.set_config(self.config, self.model_topics)
        for file in self.get_next_run_file_from_directory():
            check = file.load_file(return_content=False)
            pipe_rec = file.get_content()
            new_presentation_doc = self.apply_models_for_document_results(pipe_rec)
            pipe_rec.presentation_doc = new_presentation_doc
            self.pipeline_record_ids.append(pipe_rec.id)
            filepath = self.export_pipeline_record_to_file(pipe_rec)
            if filepath:
                self.config['LOGGER'].info(f'saved intermediate file {pipe_rec.id} - {filepath}')
            else:
                self.config['LOGGER'].info(f'failed to save intermediate file {pipe_rec.id}')
        self.config['LOGGER'].info(f'completed text-classification processing for file {len(self.pipeline_record_ids)}')
        return True
    






#from src.models.classification import classifier
from src.models.classification import TextClassifier
import time
import json

class TextClassifyEcommEmailTask(Task):
    """Apply text classification to documents
    """

    def __init__(self, config, input, output, name_diff):
        super().__init__(config, input, output, name_diff)
        self.target_files = output

    def run(self):
        TextClassifier.config(self.config)
        intermediate_save_dir=self.target_files.directory
        unprocessed_files = self.get_next_run_file_from_directory()
        if len(unprocessed_files)>0:
            #process by batch
            for idx, batches in enumerate( utils.get_next_batch_from_list(unprocessed_files, self.config['BATCH_COUNT']) ):
                """
                batch_files = asr.run_workflow(
                    config=self.config,
                    sound_files=batch, 
                    intermediate_save_dir=self.target_files.directory,
                    infer_text_classify_only=False
                    )
                self.config['LOGGER'].info(f"end model workflow, batch-index: {idx} with {len(batch_files)} files")

                """
                #run classification models on each: chunk,item
                dialogues = []
                for idx, batch in enumerate(batches):
                    dialogue = File(filepath=batch, filetype='pickle').load_file(return_content=True)
                    #with open(batch, 'r') as f_in:
                    #    dialogue = json.load(f_in)
                    #dialogues[idx]['classifier'] = []
                    dialogue['classifier'] = []
                    for chunk in dialogue['chunks']:
                        results = TextClassifier.run(chunk)
                        for result in results:
                            if result != None:
                                dialogue['classifier'].append(result)
                            else:
                                dialogue['classifier'].append({})
                    dialogue['time_textmdl'] = time.time() - self.config['START_TIME']
                    dialogues.append(dialogue)
                    self.config['LOGGER'].info(f'text-classification processing for file {idx} - {dialogue["file_name"]}')
                
        #save
        from src.io import export

        save_json_paths = []
        if intermediate_save_dir:
            for idx, dialogue in enumerate(dialogues):
                save_path = Path(intermediate_save_dir) / f'{dialogue["file_name"]}.json'
                try:
                    with open(save_path, 'w') as f:
                        json.dump(dialogue, f)
                    save_json_paths.append( str(save_path) )
                    self.config['LOGGER'].info(f'saved intermediate file {idx} - {save_path}')
                except Exception as e:
                    print(e)
                #TODO:   dialogues.extend(processed_dialogues)   #combine records of previously processed dialogues

        return save_json_paths





#from src.models.classification import classifier
from src.models.classification import TextClassifier
import time
import json

class TextClassifyEcommTask(Task):
    """Apply text classification to documents
    """

    def __init__(self, config, input, output, name_diff):
        super().__init__(config, input, output, name_diff)
        self.target_files = output

    def run(self):
        TextClassifier.config(self.config)
        intermediate_save_dir=self.target_files.directory
        unprocessed_files = self.get_next_run_file_from_directory()
        if len(unprocessed_files)>0:
            #process by batch
            for idx, batches in enumerate( utils.get_next_batch_from_list(unprocessed_files, self.config['BATCH_COUNT']) ):
                """
                batch_files = asr.run_workflow(
                    config=self.config,
                    sound_files=batch, 
                    intermediate_save_dir=self.target_files.directory,
                    infer_text_classify_only=False
                    )
                self.config['LOGGER'].info(f"end model workflow, batch-index: {idx} with {len(batch_files)} files")

                """
                #run classification models on each: chunk,item
                dialogues = []
                for idx, batch in enumerate(batches):
                    dialogue = File(filepath=batch, filetype='pickle').load_file(return_content=True)
                    #with open(batch, 'r') as f_in:
                    #    dialogue = json.load(f_in)
                    #dialogues[idx]['classifier'] = []
                    dialogue['classifier'] = []
                    for chunk in dialogue['chunks']:
                        results = TextClassifier.run(chunk)
                        for result in results:
                            if result != None:
                                dialogue['classifier'].append(result)
                            else:
                                dialogue['classifier'].append({})
                    dialogue['time_textmdl'] = time.time() - self.config['START_TIME']
                    dialogues.append(dialogue)
                    self.config['LOGGER'].info(f'text-classification processing for file {idx} - {dialogue["file_name"]}')
                
        #save
        from src.io import export

        save_json_paths = []
        if intermediate_save_dir:
            for idx, dialogue in enumerate(dialogues):
                save_path = Path(intermediate_save_dir) / f'{dialogue["file_name"]}.json'
                try:
                    with open(save_path, 'w') as f:
                        json.dump(dialogue, f)
                    save_json_paths.append( str(save_path) )
                    self.config['LOGGER'].info(f'saved intermediate file {idx} - {save_path}')
                except Exception as e:
                    print(e)
                #TODO:   dialogues.extend(processed_dialogues)   #combine records of previously processed dialogues

        return save_json_paths
'''


from src.modules.model_ensemble.AudioTranscription import AudioTranscription

class AsrTask(Task):
    """Apply Automatic Speech Recognition to audio files
    """

    def __init__(self, config, input, output):
        super().__init__(config, input, output)
        self.target_files = output
        self.asr = AudioTranscription('<name>', config)
        #self.infer_text_classify_only = False       #TODO:separate into another Task???

    def run(self):
        unprocessed_files = self.get_next_run_file_from_directory()
        if len(unprocessed_files)>0:
            #process by batch
            for idx, batch in enumerate( utils.get_next_batch_from_list(unprocessed_files, self.config['BATCH_RECORD_COUNT']) ):
                batch_files = self.asr.run(
                    sound_files=batch, 
                    intermediate_save_dir=self.target_files.directory,
                    infer_text_classify_only=False
                    )
                self.config['LOGGER'].info(f"end model workflow, batch-index: {idx} with {len(batch_files)} files")
        return True