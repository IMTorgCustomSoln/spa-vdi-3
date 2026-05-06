import { pipeline } from "@huggingface/transformers";
import { RecursiveCharacterTextSplitter } from "./langchain_mimic";

/**
 * You can try different models:
 * @link https://huggingface.co/models?pipeline_tag=feature-extraction&library=transformers.js

export const modelNames = [
  'Xenova/all-MiniLM-L6-v2',
  'Supabase/gte-small',
  'mixedbread-ai/mxbai-embed-large-v1',
  'jinaai/jina-embeddings-v2-base-zh',
  'Xenova/paraphrase-multilingual-mpnet-base-v2',
  'jinaai/jina-embeddings-v2-base-code',
  'Xenova/multilingual-e5-large',
  'WhereIsAI/UAE-Large-V1',
  'jinaai/jina-embeddings-v2-base-de',
  'jinaai/jina-embeddings-v2-base-en'
];
export const DEFAULT_MODEL_NAME = '';
 */

let summarizer = null;

self.onmessage = async (event) => {
    const { text } = event.data;

    try {
        if (!summarizer){
            self.postMessage({ status: 'init', message: 'Loading model...' });
            const hasWebGpu = !!navigator.gpu
            const summarizer = await pipeline(
                'summarization',
                'Xenova/distilbart-cnn-6-6',
                {
                    device: hasWebGpu ? 'webgpu' : 'wasm',
                    dtype: 'fp32', 
                })
            const splitter = new RecursiveCharacterTextSplitter({
                chunkSize: 1000,
                chunkOverlap: 200
            })
            const docs = await splitter.createDocuments([text])
            let chunkSummaries = []
            for (let i=0; i<docs.length; i++){
                self.postMessage(
                    {
                        status: 'processing', 
                        current: i+1,
                        total: docs.length 
                    })
                const res = await summarizer(docs[i].pageContent, {
                    max_new_tokens: 100,
                })
                chunkSummaries.push(res[0].summary_text)
            }
            const finalSummary = chunkSummaries.join(' ')
            self.postMessage({status: 'complete', summary: finalSummary})
        }
    } catch (error) {
        self.postMessage({ status: 'error', error: error.message})
    }
}