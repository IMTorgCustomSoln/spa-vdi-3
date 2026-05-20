//import { FeatureExtractionPipeline, pipeline, env } from "@huggingface/transformers";
import { getFromMapOrCreate } from 'rxdb/plugins/core';
import { getVectorFromText } from './text-embed-function';


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
export const DEFAULT_MODEL_NAME = modelNames[0];

/*
const pipePromises = new Map();

export let extractor = null;
export async function getExtractor() {
    if (!extractor) {
        env.allowLocalModels = false;
        env.useBrowserCache = false;
        extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    }
    return extractor;
};
*/

self.onmessage = async (e) => {
  //debugger
  if('data' in e.data){
    try {
        //const { data } = e.data; // e.data.data is the text from WorkerPool.run()
        const id = e.data.id
        const text = e.data.data.text

        //const pipe = await getExtractor();
        //env.allowLocalModels = true;
        //env.useBrowserCache = true;
        //extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
        // Run inference
        const output = await getVectorFromText(text)

        /* Send back the result (output.data is a Float32Array)
        self.postMessage({
            success: true,
            result: {
              page: data.page, 
              index: data.index,
              text: data.text,
              embedding: output.data
            },
            error: null
        })
        //}, [output.data.buffer]);
    } catch (error) {
        self.postMessage({
            success: false,
            error: error.message
        });
    */
      const vectorItem = e.data.data
      vectorItem['embedding'] = output
      self.postMessage({id, res: vectorItem});
    } catch (error) {
      self.postMessage({id, res: null, error: error.messag });
    }

    }
};

/*
self.onmessage = async (e) => {
  if(e.data.data){
    try{
      const result = await getVectorFromText(e.data.data, DEFAULT_MODEL_NAME);
      const transferables = result instanceof ArrayBuffer ? [result] : []
      self.postMessage({success: true, result}, transferables);
    } catch (error){
      self.postMessage({success: false, error: error.message})
    }
  }
};
*/

/*
export async function getVectorFromText(text, modelName){
  env.allowLocalModels = false
  env.useBrowserCache = false
  const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2')
  const output = await extractor(text, {pooling: "mean", normalize: true})
  const embedding = Array.from(output.data)
  return embedding
}

export async function getVectorFromText(text, modelName) {
  const pipe = await getFromMapOrCreate(
    pipePromises,
    modelName,
    () => pipeline(
      "feature-extraction",
      modelName
    )
  );
  const output = await pipe(text, {
    pooling: "mean",
    normalize: true,
  });
  const embedding = Array.from(output.data);
  return embedding;
}
*/