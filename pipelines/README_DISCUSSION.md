# Discussion on foundation Task class

### Functionality:

* `get_next_run_files()`
* ~~`run_logic_on_individual_file(func)`~~
* ~~`run_logic_on_file_batches(func)`~~
* `run_logic_on_indexed_files(func)`
* `run_logic_on_batch_of_indexed_files(func)`

TODO: index = [{idx-1:['path/to/single/file.txt']}]

### Crux: 

* from many different input formats
  - individual files
  - single file with many files grouped in an index
* create a single pipeline
  - handles logging, errors, batching, ...
  - ???
* of sequenced, interchangeable task-components that work on individual records

this was the problem with spacy: not very flexible in data format and shape


### Explanation:

Text is often worked on (processed), individually.  There is a nice CS term: ridiculously parallel.  Unfortunately, that assumes all text chunks are independent.  So, if you want to break that independence, such as working on a document, or maybe information from a group of documents, then things start to get messy.  To fix this, you can add metadata for each chunk.  But, there are still many complications.
