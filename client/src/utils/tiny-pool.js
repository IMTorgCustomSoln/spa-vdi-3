

class TinyPool {
    constructor(worker, size=4){
        this.workers = Array.from({length: size}, () => new worker(), { type: 'module'});
        this.reqs = new Map();
        this.id = 0;

        //setup permanent listeners
        this.workers.forEach(w => w.onmessage = e => {
            const {id, res, error } = e.data;
            const resolver = this.reqs.get(id);

            if (resolver) {
                this.reqs.delete(id);
                if (error) console.log(`Worker error: ${error}`);
                resolver(res);
            }
        });
    }
    run(text) {
        return new Promise(res => {
            const id = this.id++;
            this.reqs.set(id, res);
            this.workers[id % this.workers.length].postMessage({ id, data: text});
        });
    }
}

import TextWorker from './text-embed-worker.js?worker';
export const embeddingPool = new TinyPool(TextWorker);