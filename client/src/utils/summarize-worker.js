import { pipeline } from "@huggingface/transformers";
import { RecursiveCharacterTextSplitter } from "@/utils/langchain_mimic.js";
import { summaryModel } from "./summarize-function";

self.onmessage = async (event) => {
    const {id, message}  = event.data;
    const text = message

    try {
        if (!summaryModel.generator){
            //self.postMessage({ status: 'init', message: 'Loading model...' });
            const check_initialized = await summaryModel.initialize()
        }
        const splitter = new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
            chunkOverlap: 200
        })
        const docs = await splitter.splitText(text)
        let chunkSummaries = []
        if (docs.length > 1){
            for (let i=0; i<docs.length; i++){
                console.log({status: 'summaryWorker processing long document', current: i+1, total: docs.length })
                let chunk = docs[i]
                const res = await summaryModel.summarizeText(chunk)
                chunkSummaries.push(res[0].summary_text)
            }
        } else {
            chunkSummaries.push(docs[0])
        }
        const combineSummaries = chunkSummaries.join(' ')
        const finalRes = await summaryModel.summarizeText(combineSummaries)
        self.postMessage({status: 'complete', content: finalRes})
    } catch (error) {
        self.postMessage({ status: 'error', content: null, error: error.message})
    }
}