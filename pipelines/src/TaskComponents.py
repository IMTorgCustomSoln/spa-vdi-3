#!/usr/bin/env python3
"""
Task class


TODO: apply `get_next_run_files()` to new Tasks
  - but instead of simple review of files, apply to content of files
  - ensure it picks-up at last url - it is currently only designed for last file
"""

__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "AGPL-3.0"


from src.Files import File
from src.io import export
from src.io import utils
from src.modules.enterodoc.entero_document.url import UrlFactory#, UrlEncoder
from src.Task import Task

from pathlib import Path
import sys
import datetime






'''
from src.io import utils

class UnzipTask(Task):
    """Decompress archive files in a folder"""

    def __init__(self, config, input, output):
        super().__init__(config, input, output)
        self.target_folder = output.directory
        self.target_extension=['.wav','.mp3','.mp4']

    def run(self):
        sound_files_list = []
        for file in self.get_next_run_file():
            extracted_sound_files = utils.decompress_filepath_archives(
                filepath=file.filepath,
                extract_dir=self.target_folder,
                target_extension=self.target_extension
                )
            sound_files_list.extend(extracted_sound_files)
        sound_files_list = [file for file in set(sound_files_list) if file!=None]
        self.config['LOGGER'].info(f"end ingest file location from {self.input_files.directory.resolve().__str__()} with {len(sound_files_list)} files matching {self.target_extension}")
        return True
'''

from src.models import asr

class AsrTask(Task):
    """Apply Automatic Speech Recognition to audio files
    """

    def __init__(self, config, input, output, name_diff):
        super().__init__(config, input, output, name_diff)
        self.target_files = output
        #self.infer_text_classify_only = False       #TODO:separate into another Task???

    def run(self):
        unprocessed_files = self.get_next_run_file()
        if len(unprocessed_files)>0:
            #process by batch
            for idx, batch in enumerate( utils.get_next_batch_from_list(unprocessed_files, self.config['BATCH_COUNT']) ):
                batch_files = asr.run_workflow(
                    config=self.config,
                    sound_files=batch, 
                    intermediate_save_dir=self.target_files.directory,
                    infer_text_classify_only=False
                    )
                self.config['LOGGER'].info(f"end model workflow, batch-index: {idx} with {len(batch_files)} files")
        return True
    

#from src.models.classification import classifier
from src.models.classification import TextClassifier
import time
import json

class TextClassificationTask(Task):
    """Apply text classification to documents
    """

    def __init__(self, config, input, output, name_diff):
        super().__init__(config, input, output, name_diff)
        self.target_files = output

    def run(self):
        TextClassifier.config(self.config)
        intermediate_save_dir=self.target_files.directory
        unprocessed_files = self.get_next_run_file()
        if len(unprocessed_files)>0:
            #process by batch
            for idx, batches in enumerate( utils.get_next_batch_from_list(unprocessed_files, self.config['BATCH_COUNT']) ):
                '''
                batch_files = asr.run_workflow(
                    config=self.config,
                    sound_files=batch, 
                    intermediate_save_dir=self.target_files.directory,
                    infer_text_classify_only=False
                    )
                self.config['LOGGER'].info(f"end model workflow, batch-index: {idx} with {len(batch_files)} files")

                '''
                #run classification models on each: chunk,item
                dialogues = []
                for idx, batch in enumerate(batches):
                    #dialogue = File(filepath=batch, type='json').load_file(return_content=True)
                    with open(batch, 'r') as f_in:
                        dialogue = json.load(f_in)
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


class ExportAsrToVdiWorkspaceTask(Task):
    """Export files to a VDI Workspace file."""

    def __init__(self, config, input, output):
        super().__init__(config, input, output)
        self.target_folder = output.directory

    def run(self):
        #load schema
        workspace_filepath = self.config['WORKING_DIR'] / 'workspace_schema_v0.2.1.json'
        if workspace_filepath.is_file():
            self.config['WORKSPACE_SCHEMA'] = File(workspace_filepath, 'schema').load_file(return_content=True)
            self.config['LOGGER'].info(f"workspace schema loaded")
        else:
            self.config['LOGGER'].info(f"run prepare() to load workspace schema")
            sys.exit()

        #export by batch
        processed_files = self.get_next_run_file()
        cnt = 0
        for idx, batch in enumerate( utils.get_next_batch_from_list(processed_files, self.config['BATCH_COUNT']) ):
            self.config['LOGGER'].info("begin export")
            batch_span = f'{cnt+idx}-{len(batch)}'
            dt = datetime.datetime.now().isoformat().split('T')[0].replace('-','')
            export_filepath = self.target_folder / f'VDI_ApplicationStateData_v0.2.1_{dt}_{batch_span}.gz'    #v0.2.1_YYYYMMDD_0-100.gz
            dialogues = [File(file, 'json').load_file(return_content=True)
                         for file in batch
                         ]
            check = export.export_dialogues_to_output(
                schema=self.config['WORKSPACE_SCHEMA'], 
                dialogues=dialogues, 
                filepath=export_filepath, 
                output_type='vdi_workspace'
                )
            cnt = len(batch)
            self.config['LOGGER'].info(f"Data processed for batch-{idx+1}: {check}")
        return True


from .Files import File
from src.modules.enterodoc.entero_document.url import UrlFactory, UniformResourceLocator
from .web.crawler import Crawler, empty_scenario, example_udap_search_terms

import copy


class ImportAndValidateUrlsTask(Task):
    """Import initial root URL list and validate they exist."""

    def run(self):
        URL = UrlFactory()
        crawler = Crawler(
                scenario=empty_scenario,
                logger=self.config['LOGGER'],
                exporter=None
            )
        input_files = self.get_next_run_file()
        if len(input_files) == 1:   #only one input file
            input_file = input_files[0]
            records = File(filepath=input_file, filetype='yaml').load_file(return_content=True)
            for key, item in records.items():
                item['given_urls'].insert(0, item['root_url'])
                url_list = [URL.build(url) for url in item['given_urls'] if URL.build(url) != None]
                new_scenario = copy.deepcopy(empty_scenario)
                new_scenario.urls = url_list
                crawler.add_scenario(new_scenario)
                valid_urls = crawler.check_urls_are_valid(url_list)
                valid_urls_str = [url.__repr__() for url in valid_urls]
                item['_valid_urls'] = valid_urls_str
                self.config['LOGGER'].info(f"validated {len(valid_urls)} urls and saved to target {key} ")
            #output
            outfile = self.output_files.directory / 'urls.json'
            out_file = File(filepath=outfile, filetype='json')
            out_file.content = records
            check = out_file.export_to_file()
            self.config['LOGGER'].info(f"end ingest file of {len(input_files)} files")
        else:
            self.config['LOGGER'].info(f"urls previously validated")
            check=True
        return check


class CrawlUrlsTask(Task):
    """Collect branch-and-leaf URLs from initial roots."""

    def run(self):
        def flatten_extend(nested):
            flat_list = []
            for row in nested:
                flat_list.extend(row)
            return flat_list
    
        URL = UrlFactory()
        input_files = self.get_next_run_file(method='update')
        if len(input_files) == 1:
            input_file = input_files[0]
            records = File(filepath=input_file, filetype='json').load_file(return_content=True)
            for idx, (key, item) in enumerate(records.items()):
                valid_urls = [URL.build(url) for url in copy.deepcopy(item['_valid_urls'])]
                #prepare scenario
                new_scenario = copy.deepcopy(empty_scenario)
                new_scenario.base_url = URL.build(item['root_url'])
                new_scenario.urls = valid_urls
                new_scenario.list_of_search_terms = example_udap_search_terms
                new_scenario.number_of_search_results = 20
                new_scenario.depth = 2
                #use crawler
                crawler = Crawler(
                    scenario=new_scenario,
                    logger=self.config['LOGGER'],
                    exporter=None
                    )
                crawler.scenario._valid_urls = valid_urls
                result_urls = crawler.generate_href_chain()
                flat_urls = flatten_extend(list(result_urls.values()) )
                combined_urls = list(set(flat_urls))
                record = {}
                record[key] = item
                record[key]['_result_urls'] = combined_urls
                self.config['LOGGER'].info(f"searched { len(list(result_urls.keys())) } root urls to find leaf results of {len(combined_urls)} urls and saved to target {key} ")
                #output
                outfile = self.output_files.directory / f'urls{idx}-{key}.json'
                out_file = File(filepath=outfile, filetype='json')
                out_file.content = record
                check = out_file.export_to_file()
                self.config['LOGGER'].info(f"end ingest file of {len(input_files)} files")
        else:
            self.config['LOGGER'].info(f"urls previously crawled")
            check=True
        return check



from src.modules.enterodoc.entero_document.config import ConfigObj
from src.modules.enterodoc.entero_document.document_factory import DocumentFactory
from src.modules.enterodoc.entero_document.record import DocumentRecord

class ConvertUrlDocToPdf(Task):
    """Download URL document to memory then convert to PDF Format."""

    def run(self):
        #input
        check = None
        URL = UrlFactory()
        ConfigObj.set_logger(self.config['LOGGER'])
        Doc = DocumentFactory(ConfigObj)
        docrec = DocumentRecord()
        input_files = self.get_next_run_file(method='update')
        if len(input_files) > 0:
            for file_idx, file in enumerate(input_files):
                record = File(filepath=file, filetype='json').load_file(return_content=True)
                #process
                for key, item in record.items():
                    for url_idx, url_str in enumerate(item['_result_urls']   ):     #<<< TODO:remove slice [:3]
                        url = URL.build(url_str)
                        check = url.run_data_requests_()
                        if not check: 
                            raise Exception('there was an error')
                        doc = Doc.build(url)
                        try:
                            check = docrec.validate_object_attrs(doc)
                            record_dict = doc.get_record()
                            #format
                            #TODO:use pickle to keep all data
                            #TODO:make DocumenRecord able to be pickled, ref: https://stackoverflow.com/questions/2049849/why-cant-i-pickle-this-object
                        except Exception as e:
                            self.config['LOGGER'].info(f"DocumentRecord attribute validation error with url: {url_str}")
                            self.config['LOGGER'].info(e)
                            continue
                        #output
                        outfile = self.output_files.directory / f'doc-{key}-{url_idx}.json'
                        out_file = File(filepath=outfile, filetype='json')
                        out_file.content = record_dict
                        check = out_file.export_to_file()
                        #log url
                        #log file: self.config['LOGGER'].info(f"end ingest file of {len(input_files)} files")
                        #log Task: self.config['LOGGER'].info(f"validated {len(results)} urls and saved to location {self.output_files.directory.__str__()} ")
        else:
            self.config['LOGGER'].info(f"urls previously converted")
            check = True
        return check



from src.models.classification import TextClassifier

class ApplyModelsTask(Task):
    """Apply models to each doc record text."""

    def run(self):
        TextClassifier.config(self.config)
        input_files = self.get_next_run_file()
        if len(input_files) > 0:
            for file in input_files:
                record = File(filepath=file, filetype='json').load_file(return_content=True)
                record['classifier'] = []
                #process
                chunks = []
                for idx, (page, text) in enumerate(record['body_pages'].items()):
                    str_utf8 = text.encode('ascii', 'ignore').decode('utf-8').replace('\n',' ')
                    str_clean = ' '.join(str_utf8.split())
                    sentences = [{'text': item} for item in str_clean.split('.')]
                    chunks.extend(sentences)
                for chunk in chunks:
                    results = TextClassifier.run(chunk)
                    for result in results:
                        if result != None:
                            record['classifier'].append(result)
                        else:
                            record['classifier'].append({})
                record['time_textmdl'] = time.time() - self.config['START_TIME']
                #output
                outfile = self.output_files.directory / file.name
                out_file = File(filepath=outfile, filetype='json')
                out_file.content = record
                check = out_file.export_to_file()
            self.config['LOGGER'].info(f"end ingest file of {len(input_files)} files")
            #self.config['LOGGER'].info(f"validated {len(results)} urls and saved to location {self.output_files.directory.__str__()} ")
        else:
            self.config['LOGGER'].info(f"urls previously validated")
            check=True
        return check
    

class ExportVdiWorkspaceTask(Task):
    """Export files to a VDI Workspace file."""

    def __init__(self, config, input, output):
        super().__init__(config, input, output)
        self.target_folder = output.directory

    def run(self):
        #load schema
        workspace_filepath = self.config['WORKING_DIR'] / 'workspace_schema_v0.2.1.json'
        if workspace_filepath.is_file():
            self.config['WORKSPACE_SCHEMA'] = File(workspace_filepath, 'schema').load_file(return_content=True)
            self.config['LOGGER'].info(f"workspace schema loaded")
        else:
            self.config['LOGGER'].info(f"run prepare() to load workspace schema")
            sys.exit()

        #export by batch
        processed_files = self.get_next_run_file()
        processed_files.sort()
        cnt = 0
        for idx, batch in enumerate( utils.get_next_batch_from_list(processed_files, self.config['BATCH_COUNT']) ):
            self.config['LOGGER'].info("begin export")
            batch_span = f'{cnt+idx}-{len(batch)}'
            dt = datetime.datetime.now().isoformat().split('T')[0].replace('-','')
            export_filepath = self.target_folder / f'VDI_ApplicationStateData_v0.2.1_{dt}_{batch_span}.gz'    #v0.2.1_YYYYMMDD_0-100.gz
            #dialogues
            if False:
                documents = [File(file, 'json').load_file(return_content=True)
                         for file in batch
                         ]
                check = export.export_dialogues_to_output(
                    schema=self.config['WORKSPACE_SCHEMA'], 
                    records=documents, 
                    filepath=export_filepath,
                    output_type='vdi_workspace'
                    )
            #documents
            if False:
                file = batch[0]
                documents = File(file, 'json').load_file(return_content=True)
                check = export.export_documents_to_vdiworkspace(
                    schema=self.config['WORKSPACE_SCHEMA'], 
                    records=documents, 
                    filepath=export_filepath
                    )
            #document batch
            if True:
                documents = [File(file, 'json').load_file(return_content=True)
                             for file in batch
                             ]
                check = export.new_site_scrape_export(
                    schema=self.config['WORKSPACE_SCHEMA'], 
                    documents=documents, 
                    filepath=export_filepath,
                    output_type='vdi_workspace'
                    )
            cnt = len(batch)
            self.config['LOGGER'].info(f"Data processed for batch-{idx+1}: {check}")
        return True
    

from src.io.export import uint8array_to_pdf_file
import pandas as pd

class ExportIndividualPdfTask(Task):
    """Export files to individual PDF files."""

    def __init__(self, config, input, output):
        super().__init__(config, input, output)
        self.target_folder = output.directory

    def run(self):
        #export by batch
        processed_files = self.get_next_run_file()
        processed_files.sort()
        export_fields = ['id', 'date', 'page_nos', 'filetype', 'file_extension', 'file_size_mb', 'title', 'filename_original', 'filepath', 'pp_toc']
        report_records = []
        cnt = 0
        for file in processed_files:
            #import and convert
            self.config['LOGGER'].info("begin export")
            record = File(filepath=file, filetype='json').load_file(return_content=True)
            pdf_bytes = uint8array_to_pdf_file(record["file_uint8arr"])
            #export
            outfile = self.output_files.directory / f'{file.stem}.pdf'
            #out_file = File(filepath=outfile, type='pdf')
            #out_file.content = pdf_bytes
            #check = out_file.export_to_file()
            with open(outfile, 'wb') as f:
                f.write(pdf_bytes)
            check = True
            if check:
                cnt = cnt + 1
                record['printed'] = True
            else:
                record['printed'] = False
            report_records.append(record)
        df = pd.DataFrame(report_records)
        report = df[export_fields]
        reportfile = self.output_files.directory / 'report.csv'
        report.to_csv(reportfile)
        self.config['LOGGER'].info(f"end export {cnt} of {len(processed_files)} files")
        return True
    








from src.modules.parse_emails.parse_emails import EmailParser

class ImportEmailsTask(Task):
    """Import email files and validate quality.
    
    ecomms types:
    * email - .eml, .msg, .pst, .mbox, ...
    """

    def __init__(self, config, input, output):
        super().__init__(config, input, output)
        self.target_folder = output.directory
        self.target_extension=['.eml','.msg']

    def run(self):
        #create messages
        check = True
        ecomms_files_list = []
        for file in self.get_next_run_file():
            try:
                email_parser = EmailParser(file_path=file, max_depth=2)
                results = email_parser.parse()
                if type(results) == list:
                    ecomms_files_list.extend(results)
                elif type(results) == dict:
                    ecomms_files_list.append(results)
            except:
                pass
        ecomms_validated_files_list = [file for file in ecomms_files_list if file!=None]
        #output
        for file in ecomms_validated_files_list:
            outfile = self.output_files.directory / f"{file['FileName']}.pickle"
            out_file = File(filepath=outfile, filetype='pickle')
            out_file.content = file
            check = out_file.export_to_file()
        self.config['LOGGER'].info(f"end ingest file location from {self.input_files.directory.resolve().__str__()} with {len(ecomms_validated_files_list)} files matching {self.target_extension}")
        return check