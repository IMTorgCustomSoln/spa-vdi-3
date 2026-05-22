//import { pipeline } from "@huggingface/transformers";
//import { getFromMapOrCreate } from 'rxdb/plugins/core';
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

/*
const pipePromises = new Map();
const model_id = 'onnx-community/SmolLM2-135M-Instruct'

export async function summarizeText(text){
    const hasWebGpu = !!navigator.gpu
    const summarizer = await pipeline(
        'summarization',
        model_id,
        //'Xenova/distilbart-cnn-6-6',
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
*/

import { pipeline, env} from "@huggingface/transformers";


//force the library to use the browser cache api instead of local asset files
env.allowLocalModels = false;
//critical: ensure local development ('/') and production assets match the same cache origin
env.useBrowserCacheURL = window?.location?.origin || self?.location?.origin || '/';


class SummaryModel {

    constructor(){
        this.model = 'Xenova/Qwen1.5-0.5B-Chat';
        this.generator = null;
        this.tokenizer = null;
    }

    async initialize(){
        //const model_id = 'Xenova/Qwen1.5-0.5B-Chat'//'onnx-community/SmolLM2-135M-Instruct'
        //let checkCached = isModelCached(model_id)
        //const use_cache = true;
        //env.allowLocalModels = use_cache;
        env.useBrowserCache = true;
        if(!this.generator){
            this.generator = await pipeline('text-generation', this.model_id, {
                device: 'webgpu',
                quantized: true,
                dtype: 'q4'//'fp16',
            });
        }
        /*
        if(!this.tokenizer){
            this.tokenizer = await AutoTokenizer.from_pretrained(this.model_id);
        }*/
        return true
    }

    async summarizeText(message) {
        const output = await this.generator(message, {
            max_new_tokens: 100,
            temperature: 0.7,
            return_full_text: false
        });
        const result = output[0].summary_text
        return result
    };
}

export const summaryModel = new SummaryModel()