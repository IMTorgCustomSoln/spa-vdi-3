
export class WorkerPool{

    constructor(workerPath, poolSize){
        this.poolSize = poolSize || navigator.hardwareConcurrency || 4;
        this.timeout = 100000;
        this.maxRetries = 2;
        this.scriptPath = new URL(workerPath, import.meta.url)

        this.workers = []
        this.activeWorkers = new Set()
        this.queue = []

        this.init()
    }

    init(){
        for(let i=0; i< this.poolSize; i++){
            this.createWorker()
        }
    }
    createWorker(){
        const worker = new Worker(this.scriptPath, { type: 'module' })
        worker.onmessage = (e) => this.handleResponse(worker, e)
        worker.onerror = () => this.handleDeath(worker)
        this.workers.push(worker)
        console.log(this.scriptPath)
    }
    run(data){
        return new Promise((resolve, reject)=>{
            this.queue.push({data, resolve, reject, retries:0})
            this.processNext()
        })
    }
    processNext(){
        if(this.queue.length === 0) return;

        const worker = this.workers.find(w => !this.activeWorkers.has(w));
        if(!worker) return;

        const task = this.queue.shift()
        this.activeWorkers.add(worker)
        worker.currentTask = task

        task.timer = setTimeout(() => this.handleDeath(worker, 'Timeout'))

        //const transfer = data => (data instanceof ArrayBuffer) ? [data] : data
        //const transferables = (task.data instanceof ArrayBuffer) ? [task.data] : []
        //worker.postMessage({data: task.data}, transferables)
        worker.postMessage(task.data)
    }
    handleResponse(worker, { success, result, error}){
        const task = worker.currentTask
        clearTimeout(task.timer)
        this.activeWorkers.delete(worker)

        if(success){
            task.resolve(result)
        } else {
            this.attemptRetry(task, error)
        }
        this.processNext()
    }
    handleDeath(worker, reason = 'Crash'){
        const task = worker.currentTask
        if(task) clearTimeout(task.timer)

        worker.terminate()
        
        // -- FIX: Replace the dead worker --
        this.workers = this.workers.filter(w => w !== worker)
        this.activeWorkers.delete(worker)
        this.createWorker() 
        // ---------------------------------

        if(task){
            this.attemptRetry(task, reason)
        }
        this.processNext()
    }
    attemptRetry(task, error){
        if(task.retries < this.maxRetries){
            task.retries++
            this.queue.unshift(task)
        } else {
            task.reject(`Task failed after ${task.retries} retries. Error ${error}`)
        }
    }
}