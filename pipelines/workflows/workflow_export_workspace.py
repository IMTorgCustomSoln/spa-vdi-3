#!/usr/bin/env python3
"""
WorkflowExportWorkspace


UseCase-1: use this for exporting email records to a VDI Workspace file
* load record files
* xform into presentation document
* export to workspace file
"""

__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "AGPL-3.0"


from src.Workflow import WorkflowNew
from src.Files import File, Files

from src.TaskImport import ImportFromLocalFileCustomFormatTask  
from src.TaskTransform import CreatePresentationDocument
from src.TaskExport import ExportToVdiWorkspaceTask, ExportBatchToVdiWorkspaceTask

from src.io import load
#TODO: from tests.estimate_processing_time import ProcessTimeQrModel

from config._constants import (
    logging_dir,
    logger
)

from pathlib import Path
import time
import sys


class WorkflowExportWorkspace(WorkflowNew):
    """..."""

    def __init__(self):
        CONFIG = {}
        try:
            #user input
            CONFIG['INPUT_DIR'] = Path('./tests/test_wf_export_workspace/data/')
            CONFIG['WORKING_DIR'] = Path('./tests/test_wf_export_workspace/tmp/')
            CONFIG['OUTPUT_DIRS'] = [Path('./tests/test_wf_export_workspace/tmp/OUTPUT')]

            #system input
            CONFIG['START_TIME'] = None
            CONFIG['LOGGER'] = logger
            CONFIG['BATCH_RECORD_COUNT'] = 50
            CONFIG['WORKSPACE_SCHEMA'] = None

            #working dirs
            CONFIG['WORKING_DIR'].mkdir(parents=True, exist_ok=True)
            DIR_VALIDATED = CONFIG['WORKING_DIR'] / '1_VALIDATED'
            DIR_XFORM = CONFIG['WORKING_DIR'] / '2_XFORM'
            DIR_OUTPUT = CONFIG['WORKING_DIR'] / '3_OUTPUT'
            self.config = CONFIG

            #files
            input_files = Files(
                name='input',
                directory_or_list=self.config['INPUT_DIR'],
                extension_patterns=['.json']
                )
            validated_files = Files(
                name='validated',
                directory_or_list=DIR_VALIDATED,
                extension_patterns=['.pickle']
                )
            xform_files = Files(
                name='xform',
                directory_or_list=DIR_XFORM,
                extension_patterns=['.pickle']
                )
            output_files = Files(
                name='output',
                directory_or_list=DIR_OUTPUT,
                extension_patterns=['.gz']
                )
            self.files = {
                'input_files': input_files,
                'validated_files': validated_files,
                'xform_files': xform_files,
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
            #tasks
            import_task = ImportFromLocalFileCustomFormatTask(
                config=self.config, 
                input=self.files['input_files'],
                output=self.files['validated_files']
                )
            xform_task = CreatePresentationDocument(
                config=self.config,
                input=self.files['validated_files'],
                output=self.files['xform_files']
                )
            output_task = ExportBatchToVdiWorkspaceTask(
                config=self.config,
                input=self.files['xform_files'],
                output=self.files['output_files'],
                vdi_schema=None
            )
            tasks = [
                import_task,
                xform_task,
                output_task
                ]
            self.tasks = tasks
        except Exception as e:
            print(e)
            sys.exit()
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
        


workflow_export_workspace = WorkflowExportWorkspace()