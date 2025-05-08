#!/usr/bin/env python3
"""
WorkflowASR
"""

__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "AGPL-3.0"


from src.Workflow import WorkflowNew
from src.Files import File, Files
from src.TaskImport import ImportFromLocalFileTask, ImportBatchDocsFromLocalFileTask 
from src.TaskTransform import (
     CreatePresentationDocument, 
    ApplyTextModelsTask
)
from src.TaskComponents import (
    UnzipTask,
    AsrTask,
    TextClassificationTask,
    ExportAsrToVdiWorkspaceTask
)
from src.Report import (
    TaskStatusReport,
    MapBatchFilesReport,
    ProcessTimeAnalysisReport
)
from src.models import prepare_models
from src.io import load
from tests.test_wf_asr.estimate_processing_time import ProcessTimeQrModel

from config._constants import (
    logging_dir,
    logger
)

from pathlib import Path
import json
import time
import sys


class WorkflowASR(WorkflowNew):
    """..."""

    def __init__(self):
        CONFIG = {}
        try:
            #user input
            CONFIG['INPUT_DIR'] = Path('./tests/test_wf_asr/data/samples/')
            CONFIG['TRAINING_DATA_DIR'] = Path('./tests/test_wf_asr/data/model_training/')     #Path('./src/data/covid/')
            CONFIG['WORKING_DIR'] = Path('./tests/test_wf_asr/tmp/')
            CONFIG['OUTPUT_DIRS'] = [Path('./tests/test_wf_asr/tmp/OUTPUT')]

            #system input
            CONFIG['START_TIME'] = None
            CONFIG['LOGGER'] = logger
            CONFIG['BATCH_COUNT'] = 25
            CONFIG['WORKSPACE_SCHEMA'] = None
            #CONFIG['REGEX_INPUT_FILES_NAMES'] = '_Calls_'
            
            #working dirs
            CONFIG['WORKING_DIR'].mkdir(parents=True, exist_ok=True)
            DIR_UNZIPPED = CONFIG['WORKING_DIR'] / '1_UNZIPPED'
            DIR_PROCESSED = CONFIG['WORKING_DIR'] / '2_PROCESSED'
            DIR_CLASSIFIED = CONFIG['WORKING_DIR'] / '3_CLASSIFIED'
            DIR_OUTPUT = CONFIG['WORKING_DIR'] / '4_OUTPUT'

            DIR_ARCHIVE = CONFIG['WORKING_DIR'] / 'ARCHIVE'
            CONFIG['DIR_ARCHIVE'] = DIR_ARCHIVE
            self.config = CONFIG

            #files
            input_files = Files(
                name='input',
                directory_or_list=self.config['INPUT_DIR'],
                extension_patterns=['.zip']
                )
            unzip_files = Files(
                name='unzip',
                directory_or_list=DIR_UNZIPPED,
                extension_patterns=['.wav','.mp3','.mp4']
                )
            processed_files = Files(
                name='processed',
                directory_or_list=DIR_PROCESSED,
                extension_patterns=['.json']
                )
            labeled_files = Files(
                name='labeled',
                directory_or_list=DIR_CLASSIFIED,
                extension_patterns=['.json']
                )
            output_files = Files(
                name='output',
                directory_or_list=DIR_OUTPUT,
                extension_patterns=['.gz']
                )
            self.files = {
                'input_files': input_files,
                'unzip_files': unzip_files,
                'processed_files': processed_files,
                'labeled_files': labeled_files,
                'output_files': output_files
            }
        except Exception as e:
            print(e)
            sys.exit()

    def config_tasks(self):
        """Configure Tasks, some of which may need to be initialized
        after other preparations.
        """
        try:
            unzip_task = UnzipTask(
                config=self.config,
                input=self.files['input_files'],
                output=self.files['unzip_files']
                )
            import_task = ImportFromLocalFileTask(
                config=self.config, 
                input=self.files['input_files'],
                output=self.files['validated_files']
                )
            #convert to document records
            #use .pickle to ensure correct type
            asr_task = AsrTask(
                config=self.config,
                input=self.files['unzip_files'],
                output=self.files['processed_files'],
                name_diff='.json'
            )
            xform_task = CreatePresentationDocument(
                config=self.config,
                input=self.files['validated_files'],
                output=self.files['xform_files']
                )
            models_task = ApplyTextModelsTask(
                config=self.config,
                input=self.files['processed_files'],
                output=self.files['labeled_files'],
            )
            """
            models_task = TextClassificationTask(
                config=self.config,
                input=self.files['processed_files'],
                output=self.files['labeled_files'],
                name_diff='.json'
            )
            """
            output_task = ExportAsrToVdiWorkspaceTask(
                config=self.config,
                input=self.files['labeled_files'],    #self.files['classified_files'],
                output=self.files['output_files']
            )
            tasks = [
                unzip_task,
                import_task,
                xform_task, 
                asr_task,
                models_task,
                output_task
                ]
            self.tasks = tasks
        except Exception as e:
            print(e)
            sys.exit()
        return True
        
    def prepare_models(self):
        """Prepare by loading train,test data and refine models"""
        self.config['LOGGER'].info("Begin prepare_models")
        check_prepare = prepare_models.finetune_classification_model()
        if not check_prepare: 
            self.config['LOGGER'].info(f"models failed to prepare")
            exit()
        self.config['LOGGER'].info("End prepare_models")
        return True

    def prepare_workspace(self):
        """Prepare workspace with output schema and file paths"""
        #prepare schema
        filepath = Path('./tests/data/VDI_ApplicationStateData_v0.2.1.gz')
        if filepath.is_file():
            workspace_schema = load.get_schema_from_workspace(filepath)
        self.config['WORKSPACE_SCHEMA'] = workspace_schema
        schema = self.config['WORKING_DIR'] / 'workspace_schema_v0.2.1.json'
        schema_file = File(schema, 'json')
        schema_file.load_file(return_content=False)
        schema_file.content = workspace_schema
        check1 = schema_file.export_to_file()
        check2 = self.config_tasks()
        return all([check1,check2])
    
    def run(self):
        """Run the workflow of tasks"""
        self.config['LOGGER'].info('begin process')
        self.config['START_TIME'] = time.time()
        for task in self.tasks:
            task.run()
        self.config['LOGGER'].info(f"end process, execution took: {round(time.time() - self.config['START_TIME'], 3)}sec")
        return True

    '''
    def report_task_status(self):
        """
        TODO:
        * typical size of files
        * outlier files that are very large
        * place code in a separate file
        """
        TaskStatusReport(
            files=self.files,
            config=self.config
        ).run()
        return True
    
    def report_map_batch_to_files(self):
        """Create .csv of files in each batch output
        """
        MapBatchFilesReport(
            files=self.files,
            config=self.config
        ).run()
        return True
    
    def report_process_time_analysis(self):
        """Analyze processing times of completed files
        """
        MapBatchFilesReport(
            files=self.files,
            config=self.config
        ).run()
        ProcessTimeAnalysisReport(
            files=self.files,
            config=self.config
        ).run()
        return True'
    '''

        


workflow_asr = WorkflowASR()