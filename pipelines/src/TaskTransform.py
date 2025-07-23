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
#from src.models.classification import TextClassifier. TODO:fix

import pandas as pd

from pathlib import Path
import time
import copy
import shutil


class UnzipTask(Task):
    """Decompress archive files in a folder"""

    def __init__(self, config, input, output):
        super().__init__(config, input, output)
        self.target_folder = output.directory
        self.target_extension=['.wav','.mp3','.mp4']

    def run(self):
        sound_files_list = []
        for file in self.get_next_run_file_from_directory():
            extracted_sound_files = utils.decompress_filepath_archives(
                filepath=file.filepath,
                extract_dir=self.target_folder,
                target_extension=self.target_extension
                )
            sound_files_list.extend(extracted_sound_files)
        sound_files_list = [file for file in set(sound_files_list) if file!=None]
        self.config['LOGGER'].info(f"end ingest file location from {self.input_files.directory.resolve().__str__()} with {len(sound_files_list)} files matching {self.target_extension}")
        return True




import ffmpeg
import subprocess

def convert_mp4_to_mp3(input_file, output_file):
    """Convert .mp4 video files to .mp3 audio files for transcription.
    
    -n `do not overwrite file if it exists`
    -i `infile`
    -q:a `q use fixed quality scale (VBR)`
    0 `...`
    -map `set input stream mapping`
    a - `...`
    
    ref: [complete list of ffmpeg flags / commands](https://gist.github.com/tayvano/6e2d456a9897f55025e25035478a3a50)
    """
    #check = ffmpeg.input(input_file).output(output_file).run()
    try:
        subprocess.run(['ffmpeg', '-n', '-i', input_file, '-q:a', '0', '-map', 'a', output_file],check=True)
        print(f"Conversion successful! Saved as {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
        return False
    except FileNotFoundError:
        print("FFmpeg is not installed or not found in your PATH.")
        return False



class FlattenFileStructureTask(Task):
    """..."""
    def __init__(self, config, input, output, convert_files=False):
        super().__init__(config, input, output)
        self.convert_files = convert_files
        self.converter_router = {
            '.mp4': {'fun': convert_mp4_to_mp3, 'outfile_suffix': '.mp3'}
        }
    def get_next_run_file_from_directory(self):
        filelist = []
        for item in list(self.input_files.directory.iterdir()):
            if item.is_dir():
                for file in list(item.iterdir()):
                    if file.is_file():
                        filelist.append(file)
        return filelist
    def run(self):
        sound_files_list = []
        for file in self.get_next_run_file_from_directory():
            if (not self.convert_files) or (file.suffix not in self.converter_router.keys()):
                outfile_path = self.output_files.directory / file.name
                shutil.copy(file.__str__(), outfile_path)
            else: 
                key = file.suffix
                outfile_suffix = self.converter_router[key]['outfile_suffix']
                outfile_path = self.output_files.directory / f'{file.stem}{outfile_suffix}'
                self.converter_router[key]['fun'](file.__str__(), outfile_path.__str__())
            sound_files_list.append(outfile_path)
        return True
   
from src.Task import PipelineRecordFactory, PipelineRecord
from itertools import groupby

class CreateSingleUrlRecordTask(Task):
    """Create a PipelineRecord from a Single File.
    The pipeline record provides the metadata and final formatted presentation document 
    for application of text models.  The `.presentation_doc` is used for final export.
    """
    def __init__(self, config, input, output):
        super().__init__(config, input, output)

    def run(self):
        factory = PipelineRecordFactory()
        for file in self.get_next_run_file_from_directory():
            check = file.load_file(return_content=False)
            doc_rec = file.get_content()
            pipe_rec = factory.create_from_id(
                id=file.get_name_only(), 
                source_type='single_url', 
                root_source=doc_rec['filepath'],
                )
            pipe_rec.collected_docs.append(doc_rec)
            check = pipe_rec.populate_presentation_doc(method='single')
            self.pipeline_record_ids.append(pipe_rec)
            filepath = self.export_pipeline_record_to_file(pipe_rec)
            self.config['LOGGER'].info(f"exported processed file to: {filepath}")
        self.config['LOGGER'].info(f"end ingest file location from {self.input_files.directory.resolve().__str__()} with {len(self.pipeline_record_ids)} files matching {self.target_extension}")
        return True
    

class CreateMultiFilelRecordTask(Task):
    """TODO:Create a PipelineRecord from a Multiple Urls.
    The pipeline record provides the metadata and final formatted presentation document 
    for application of text models.  The `.presentation_doc` is used for final export.
    """
    def __init__(self, config, input, output):
        super().__init__(config, input, output)

    def get_file_group_id(self, file):
        return file.filepath.stem.split('_')[0]
    
    def run(self):
        files = list(self.get_next_run_file_from_directory())
        files_sorted = [file for file in 
                        sorted(files, key=lambda x: self.get_file_group_id(x))
                        ]
        files_grouped = {key: list(group) for key, group in 
                         groupby(files_sorted, key=lambda x: self.get_file_group_id(x))
                         }
        factory = PipelineRecordFactory()
        for id_group, file_group in files_grouped.items():
            checks = [file.load_file(return_content=False) for file in file_group]
            source_type = 'multiple_files'
            record = factory.create_from_id(id_group, source_type)
            record.root_source = file_group[0].filepath
            record.added_sources = [file.filepath for file in file_group]
            docs = [file.get_content() for file in file_group]
            #TODO: this should be changed in the AsrTask logic
            for doc in docs: 
                doc['filetype']='audio'
                doc['filepath'] = doc['file_path']
                del doc['file_path']
                lines = [f"{chunk['timestamp']}: {chunk['text']}" for chunk in doc['chunks']]
                doc['body'] = '\n'.join(lines)
            record.collected_docs = docs
            check = record.populate_presentation_doc()
            self.pipeline_record_ids.append(record.id)
            filepath = self.export_pipeline_record_to_file(record)
            self.config['LOGGER'].info(f"exported processed file to: {filepath}")
        self.config['LOGGER'].info(f"end ingest file location from {self.input_files.directory.resolve().__str__()} with {len(self.pipeline_record_ids)} files matching {self.target_extension}")
        return True



from src.modules.enterodoc.entero_document.url import UrlFactory

class CreateMultiUrlRecordTask(Task):
    """Create a PipelineRecord from a Multiple Files.
    THIS IS NECESSARY BECAUSE:
    * URL FILES FROM `ConvertUrlDocsToPdf` ARE DIFFERENT FROM OTHER INPUT
    * MUST ACCOUNT FOR POSSIBLE MULTIPLE BANKS: `urls0-creditonebank.json`, ...
    * ONLY TAKE URLS FROM `"_result_urls": [`

    The pipeline record provides the metadata and final formatted presentation document 
    for application of text models.  The `.presentation_doc` is used for final export.
    """
    def __init__(self, config, input, output):
        super().__init__(config, input, output)

    def group_urls_by_filename(result_urls):
        """Group result_urls into a dict[filename]=[url1, url2, ...]

        #TODO:add grouping key from filename
        """
        URL = UrlFactory()
        urls_grouped_by_filename = {}
        for url in result_urls:
            Url = URL.build(url)
            filename = Url.get_filename()
            if filename not in urls_grouped_by_filename.keys():
                urls_grouped_by_filename[filename] = []
            urls_grouped_by_filename[filename].append(Url)
        return urls_grouped_by_filename

    def run(self):
        factory = PipelineRecordFactory()
        for file in self.get_next_run_file_from_directory():
            check = file.load_file(return_content=False)
            content = file.get_content()
            bank_id_key = content.keys()[0]
            urls_grouped_by_filename = self.group_urls_by_filename( content[bank_id_key]["_result_urls"] )
            for filename, group_urls in urls_grouped_by_filename.items():
                source_type = 'multiple_files'
                record = factory.create_from_id(
                    root_source=filename,
                    source_type=source_type
                )
                record.collected_docs = group_urls
                doc = group_urls[0]
                doc['filetype'] = 'url' #TODO:is this right?
                doc['filepath'] = doc['file_path']
                del doc['file_path']
                check = record.populate_presentation_doc(method='single')
                self.pipeline_record_ids.append(record.id)
                #TODO:exported filename
                filepath = self.export_pipeline_record_to_file(record)
            self.config['LOGGER'].info(f"exported processed file to: {filepath}")
        self.config['LOGGER'].info(f"end ingest file location from {self.input_files.directory.resolve().__str__()} with {len(self.pipeline_record_ids)} files matching {self.target_extension}")
        return True


class CreateSingleFileRecordTask(Task):
    """Create a PipelineRecord from a Single File.
    The pipeline record provides the metadata and final formatted presentation document 
    for application of text models.  The `.presentation_doc` is used for final export.
    """
    def __init__(self, config, input, output):
        super().__init__(config, input, output)

    def run(self):
        for file in self.get_next_run_file_from_directory():
            check = file.load_file(return_content=False)
            record = file.get_content()
            check = record.populate_presentation_doc()
            self.pipeline_record_ids.append(record.id)
            filepath = self.export_pipeline_record_to_file(record)
            self.config['LOGGER'].info(f"exported processed file to: {filepath}")
        self.config['LOGGER'].info(f"end ingest file location from {self.input_files.directory.resolve().__str__()} with {len(self.pipeline_record_ids)} files matching {self.target_extension}")
        return True
    

import fitz
from io import BytesIO
import numpy as np

class DisplayModelResultsOnPresentationDocTask(Task):
    """Prepare `.presentation_doc` .pdf with the results from models.

    TODO: this is only configured for the current test: `test_workflow_site_scrape.py`!   
    * update `config['TRAINING_DATA_DIR']:{...` so that it can incorporate options to be used, here!
    """
    def __init__(self, config, input, output):
        super().__init__(config, input, output)

    def apply_model_results_to_pdf(self, pipe_rec):
        presentation_doc = pipe_rec.presentation_doc
        nonempty_model_results = [item for item in presentation_doc['models'] if 'search' in item.keys()]
        file_uint8arr = presentation_doc['file_uint8arr']
        # highlight
        key_words = set( [item['target'] for item in nonempty_model_results if item['target']=='terms'] )
        new_uint8arr = self.style_key_words_in_uint8arr(
            file_uint8arr=file_uint8arr,
            key_words_to_match=key_words,
            method='highlight',
            span_attrs={'stroke': (0,1,0)}
            )
        # text - only terms
        key_words = set( [item['target'] for item in nonempty_model_results if item['target']!='terms'] )
        new_uint8arr = self.style_key_words_in_uint8arr(
            file_uint8arr=new_uint8arr,
            key_words_to_match=key_words,
            method='text',
            span_attrs={'font': 'Courier', 'color': (1, 0, 0)},
            #return_type='pdf_output'
            )
        presentation_doc['file_uint8arr'] = new_uint8arr
        return presentation_doc

    def style_key_words_in_uint8arr(self, file_uint8arr, key_words_to_match, method='text', span_attrs={}, return_type='uint8array'):
        """Style key words in the Uint8Array, including: highlight, font-color, etc.

        ref: https://github.com/pymupdf/PyMuPDF/discussions/3831
        ref-text style: https://github.com/pymupdf/PyMuPDF/discussions/1532
        ref-highlight: https://github.com/pymupdf/PyMuPDF/discussions/1034
        ref-text insert: https://github.com/pymupdf/PyMuPDF/discussions/2843
        """
        # checks
        default_attrs = {
            'text': {'font': 'Courier', 'fontsize': 10.0, 'color': (1, 0, 0)}, 
            'highlight': {'stroke': (0,1,0)}
            }
        return_types = ['uint8array', 'pdf_output']
        errors = []
        for key in span_attrs.keys():
            if method not in default_attrs.keys():
                errors.append(key)
        if len(errors) > 0:
            raise Exception(f'method provided, but not available for use: {errors}')
        for key in span_attrs.keys():        
            if key not in default_attrs[method]:
                errors.append(key)
        if len(errors) > 0:
            raise Exception(f'key(s) provided, but not available for use: {errors}')
        if return_type not in return_types:
            errors.append(return_type)
            raise Exception(f'return_type provided, but not available for use: {errors}')
        # apply to ocg_layer on each page
        pdf_stream = BytesIO(bytes(file_uint8arr))
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        ocg_layers = doc.layer_ui_configs()
        new_ocg_layer_name = f'layer-{method}-1'
        if ocg_layers:
            ocg_layer_names = []
            for layer_dict in ocg_layers:
                name = layer_dict.get('text', 'Unnamed Layer')
                ocg_layer_names.append(name)
                print(f"OCG Layer named {name} has properties: number- {layer_dict.get('number')} on- {layer_dict.get('on')} locked- {layer_dict.get('locked')}")
            indices = [int(layer.split('-')[2]) for layer in ocg_layer_names if method in layer]
            if len(indices) > 0:
                next_index = max( indices )
                new_ocg_layer_name = f'layer-{method}-{next_index}'
        else:
            print("No OCG layers found in the PDF.")
        current_ocg_xrf = doc.add_ocg(name = new_ocg_layer_name, on=True)
        if method == 'text':
            for page in doc:
                rects = []
                for term in key_words_to_match:
                    results = page.search_for(term)
                    rects.extend( results )
                spans = []
                for clip in rects:
                    #blocks = page.get_text("dict", clip=clip)["blocks"]
                    for block in page.get_text("dict", clip=clip)["blocks"]:
                        for ln in block["lines"]:
                            spans.extend(ln["spans"])
                #for span in spans:
                #    page.add_redact_annot(span["bbox"])
                #    page.apply_redactions()
                for span in spans:
                    # re-insert same text with new style rules
                    kwargs = {}
                    #font
                    key = 'font'
                    if key in span_attrs.keys():
                        font = fitz.Font(span_attrs[key])    # this cannot be obtained from span - only the page
                    else:
                        font = 'Courier'
                    kwargs[key] = font
                    #fontsize
                    key = 'fontsize'
                    if key in span_attrs.keys():
                        fontsize = span_attrs[key]
                    else:
                        fontsize=span["size"]
                    kwargs[key] = fontsize
                    #color
                    tw = None
                    if 'color' in span_attrs.keys():
                        tw = fitz.TextWriter(page.rect, color=span_attrs['color'])
                    else:
                        tw = fitz.TextWriter(page.rect)
                    #tw.append(span["origin"], span["text"], fontsize=span["size"], font=font)
                    tw.append(span["origin"], span["text"], **kwargs)
                    tw.write_text(page, oc=current_ocg_xrf)
        elif method == 'highlight':
            for page in doc:
                rects = []
                for term in key_words_to_match:
                    rects.extend(page.search_for(term))
                annotation=page.add_highlight_annot(rects)
                annotation.set_colors(stroke=span_attrs[key])
                annotation.set_oc(current_ocg_xrf)
                annotation.update()
        # complete results
        if return_type == return_types[0]:
            byte_stream = BytesIO()
            doc.save(byte_stream)
            doc.close()
            pdf_bytes = byte_stream.getvalue()
            byte_stream.close()
            np_array = np.frombuffer(pdf_bytes, dtype=np.uint8)
            return np_array.tolist()
        elif return_type == return_types[1]:
            doc.ez_save("x.pdf")
            doc.close()
            return True
        


        

    def run(self):
        for file in self.get_next_run_file_from_directory():
            check = file.load_file(return_content=False)
            pipe_rec = file.get_content()
            pipe_rec.presentation_doc = self.apply_model_results_to_pdf(pipe_rec)
            filepath = self.export_pipeline_record_to_file(pipe_rec)
            self.config['LOGGER'].info(f"exported processed file to: {filepath}")
        self.config['LOGGER'].info(f"end ingest file location from {self.input_files.directory.resolve().__str__()} with {len(self.pipeline_record_ids)} files matching {self.target_extension}")
        return True