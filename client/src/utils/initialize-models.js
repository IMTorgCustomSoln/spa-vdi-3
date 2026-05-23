/*
Notes:
* this should work because the models are caching files in the same place by
applying the correct hf global variables.
* also, by working with the models, directly, instead of using Workers, the
app does not have the overhead of building and terminating workers
*/
import textEmbeddingWorker  from './text-embed-worker?worker';
import chatWorker from './chat-worker?worker';
import summaryWorker from './summarize-worker?worker';

export async function initializeModels(){
    const checks = [
        initializeWorker(textEmbeddingWorker, 'textEmbeddingWorker', true),
        initializeWorker(chatWorker, 'chatWorker', false),
        //initializeWorker(summaryWorker, 'summaryWorker', false)
    ]
    const all_checks = await Promise.all(checks)
    const check = all_checks.every(item => item['status'] === true)
    if( check ){
        return true
    } else {
        return false
    }
}

async function initializeWorker(worker, workerName, nested_response){
    const text = `This is a sample text for ${workerName}.`

    //wrap the worker communication in a Promise for clean async/await usage
    function runWorkerTask(id, text, nested_response) {
        return new Promise((resolve, reject) => {

            //logic is based on message payload (nested response or not)
            if(nested_response){
                const data = {
                    id: id,
                    text: text
                }
                const myWorker = new worker()
                //listen for the specific response from this worker invocation
                myWorker.onmessage = function(event) {
                    const { id, data, error } = event.data;
                    if (!error) {
                        resolve(true);
                    } else {
                        reject(new Error(error));
                    }
                };
                myWorker.postMessage({
                    id: id,
                    data: data
                });
            } else {
                const myWorker = new worker()
                //listen for the specific response from this worker invocation
                myWorker.onmessage = function(event) {
                    const { status, content, error } = event.data;
                    if (!error) {
                        resolve(true);
                    } else {
                        reject(new Error(error));
                    }
                };
                myWorker.postMessage({
                    id: id,
                    message: text
                });
            }
        })
    }

    //run
    try {
        const result = await runWorkerTask(0, text, nested_response);
        return {worker: workerName, status: true};
    } catch (error) {
        console.error(`Worker ${workerName} failed:`, error.message);
        return {worker: workerName, status: false};
    }
}