import { FeatureExtractionPipeline, pipeline, env } from "@huggingface/transformers";
import { getFromMapOrCreate } from 'rxdb/plugins/core';



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

const pipePromises = new Map();
env.allowLocalModels = false;
let extractor = null
async function getExtractor() {
    if (!extractor) {
        // Load the feature extraction pipeline
        extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    }
    return extractor;
}

self.onmessage = async (e) => {
    try {
        const { data } = e.data; // e.data.data is the text from WorkerPool.run()

        const pipe = await getExtractor();
        
        // Run inference
        const output = await pipe(data, { pooling: 'mean', normalize: true });

        // Send back the result (output.data is a Float32Array)
        self.postMessage({
            success: true,
            result: output.data // Sending Float32Array
        });
    } catch (error) {
        self.postMessage({
            success: false,
            error: error.message
        });
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


export async function getVectorFromText(text, modelName){
  env.allowLocalModels = false
  env.useBrowserCache = false
  const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2')
  const output = await extractor(text, {pooling: "mean", normalize: true})
  const embedding = Array.from(output.data)
  return embedding
}
/*
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