import { pipeline } from "@huggingface/transformers";
import { getFromMapOrCreate } from 'rxdb/plugins/core';
//import * as transformers from "@xenova/transformers";
//import * as rxdb from "rxdb/plugins/core";

/**
 * You can try different models:
 * @link https://huggingface.co/models?pipeline_tag=feature-extraction&library=transformers.js
 */
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


const pipePromises = new Map();

export async function summarizeText(text){
    const hasWebGpu = !!navigator.gpu
    const summarizer = await pipeline(
        'summarization',
        'Xenova/distilbart-cnn-6-6',
        {
            device: hasWebGpu ? 'webgpu' : 'wasm',
            dtype: 'fp32', 
        })
    const result = await summarizer(
        text,
        {max_new_tokens:100}
    )
    return result[0].summary_text
}