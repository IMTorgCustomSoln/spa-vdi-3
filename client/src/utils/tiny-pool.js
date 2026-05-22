

class TinyPool {

    constructor(worker){
        const size = navigator.hardwareConcurrency - 1
        this.workers = Array.from({length: size}, () => new worker(), { type: 'module'});
        /*
        this.workers = Array.from({length: size}, () => {
            new Worker(
                //new worker(),
                new URL('./text-embed-worker.js?worker', import.meta.url), 
                { type: 'module'}
            )
        });*/
        this.reqs = new Map();
        this.id = 0;

        //setup permanent listeners
        this.workers.forEach(w => w.onmessage = e => {
            //debugger
            const {id, res, error } = e.data;
            const resolver = this.reqs.get(id);

            if (resolver) {
                this.reqs.delete(id);
                if (error) console.log(`Worker error: ${error}`);
                resolver(res);
            }
        });
    }
    run(vectorItem) {
        return new Promise(res => {
            const id = this.id++;
            this.reqs.set(id, res);
            this.workers[id % this.workers.length].postMessage({ id, data: vectorItem});
        });
    }
}

import TextWorker from '@/utils/text-embed-worker.js?worker';
export const embeddingPool = new TinyPool(TextWorker);
//export const embeddingPool = new TinyPool();