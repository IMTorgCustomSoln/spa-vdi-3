


## Usage

### All Workflows

* add `.env` file with the following variables: 
    - HF_TOKEN - huggingface account
    - HF_HOME - models dir
    - CURL_CA_BUNDLE - public key (.pem)
* ensure folders are populated:
    - `logs/`
    - `models_pretrained/`
    - `models_data/`
* set the following in `workflows/workflow_*.py`
    - `CONFIG['INPUT_DIR']` - location of input files
    - `CONFIG['WORKING_DIR']` - location for all files produced
* choose the workflow file from the `workflows/` directory.
* register it in the `main.py` file at `#register here`

```
python main.py workflow_* prepare_workspace
python main.py workflow_* run
```


### Automatic speech recognition, `-asr`

...


### Identifying and scraping websites, `-site_scrape`

* delete `tests/test_site_scrape/tmp/`
* set list of initial urls (one for each bank) in: `urls.txt`


### eComms discovery, `-ecomms`

...


### Text classification, `-text_classify`

...



## Create New Workflow

* use `template` workflow as guide (copy-paste the `tests/test_wf_*/` dir and `workflows/workflow_*.py` file)
* create new `workflows/workflow_*.py`
  - setup `test/test_wf_*/`
* config directories
* TODO: determine ingest with i) individual records, ii) record batches, or iii) indexed records
* add pipeline task components
  - review `tests/test_task/*` to fit class templates
* ensure current workspace output format: `tests/data/VDI_ApplicationStateData_v*.*.*.gz`
* build and test



## TODO:

* ~~create separate `workflow-text_classify`~~
* ~~enable topics of multiple models available to apply in the same pipeline~~
* ~~enable interactive work with notebook~~
* ~~create workflow template~~
* ~~convert workflow_template.py~~
* improve multi-model topics and Classification() class
  - fix vdiworkspace export so it runs with client search: pre-run models
  - ~~move pretrained_models/ to models/~~ => ERRORS, can't add .env to vscode pytest
  - ~~move src/data/ to models_data/~~
* improve interactive mode with notebook.ipynb
  - enable lists, instead of Files to be used with Tasks
  - test_workflow > test_task > test_files
* ~~create `workflow-template` to be used as a template~~
* ~~add `workflow_template` Tasks tests~~
* discussion on foundation Task class
* ~~single parent class for workflow
  - just provide i) list of tasks and ii) shape of records
  - all Files and intermediate objects created for you
  - automated validation, logging, error handling, and failover
  - use pickle to preserve objects, until task penultimate to output
  - each record should be intermediary file
* record structure for Task i/o and provisioning output
  ```urls.json
  {
  'indexed group': {
    'source': 'one_of_many',
    'root_url: 'https://www...',
    'given_urls: [Url1, Url2, ...],
    'added_docs': [DocPath1, DocPath2, ...],
    'documents': [DocumentRecord, DocumentRecord, ...]
    }
  }
  ```
* indexed group | (output format) file_field | doc_display
  - wf_site_scrape-tgt: bank_name | (vdi client) Url,DocPath | Doc
  - wf_site_scrape-multi: bank_name | (table) Url | reference to docpath
  - wf_asr: acct_num | (vdi client) acct_num | {audio file-date}\n asr_text
  - wf_ecomms: msg_chain_subject | (vdi client) msg_chain_subject-msg_count | {msg file-date}\n msg_text
  - wf_text_classify: indv_file | (indv_file) indv_file | Doc
  - wf_default: indv_file | (vdi client) indv_file | Doc


### Ecomms

* ~~output to .json~~
* ingest data
  - create proper eDiscovery class with config (include col names)
  - ~~enable mapping provided .dat column names to default set~~
  - add text msg parse to ediscovery
  - (later)setup ediscovery with .msg parse
  - ~~improve validation rules~~
  - test
  - add to Task
* orgchart
  - ~~load orgchart~~
  - integrate org chart file: just get titles
  - add to Task
* ~~prepare models~~
* export to VDI client
* (later) add visual message display


### ASR

* ~~create .csv list of files that are in each batch .gz
* ~~perform analysis on log files to estimate processing time
* integrate tests/
  - ~~test_workflow_asr.py~~
  - test_prepare_config.py
  - test_export.py
  - test_export_to_vdi_workspace.py
  - test_main.py
* maybe integrate improved whisper? english-only!, [ref: whisper-medusa](https://huggingface.co/aiola/whisper-medusa-v1)
* ensure multi-lingual whisper, [ref: multilingual](https://huggingface.co/openai/whisper-large-v3)


### Site_Scrape

* apply WorkflowNew parent 
* make modules
  - EnteroDoc
  - URL
* ~~mod  `urls.txt` and ValidateUrlsTask~~
  - ~~ValidateUrlsTask => ImportAndValidateUrlsTask~~
  - ~~change `urls.txt` to `urls.json`:~~
  ~~```urls.json
  {
  'bank_name': {
    'root_url: 'https://www...',
    'given_urls: [url1,url2, ...],
    }
  }
  ```~~
  - ~~`tmp/1_VALIDATED/urls.txt` to `valid_urls.json`~~
* mod Crawler
  - ~~`scenario.url` conflicts with crawler.check_urls_are_valid(url_list)~~
  - use consistent naming across tasks: ValidateUrlsTask, CrawlUrlsTask
  - remove: ~~generalize so 'jpmorgan' references is added to scenario~~
  - remove: ~~generalize so scenario, 'list_of_search_terms', can be changed~~
  - check why following are validated:
    + https://www.cardcenterdirect.com/
    + https://www.umb.edu/media/umassboston/editor-uploads/research/research-amp-sponsored-programs/Cost-Transfer-Policy--Procedures.pdf
* apply models
* ~~add new workflow-scraping~~
* ~~generalize classes for extensibility of workflows~~
* documentation
  - explain configuration adjustments
* ~~Workspace export~~
  - ~~no models applied~~
  - ~~record objects are not populated - only the first is repeated~~
* create reports
  - urls collected, validated, selected(reason)
  - document text results: hits, counts, snippets => VDI b-table export to excel, [ref](https://stackoverflow.com/questions/71465593/exporting-bootstrap-table-to-excel-or-pdf)
* compress pdf size with [`pdfsizeopt](https://github.com/pts/pdfsizeopt), but must update for py3.*, first