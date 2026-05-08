import { FeatureExtractionPipeline, pipeline } from "@huggingface/transformers";
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
export const DEFAULT_MODEL_NAME = modelNames[0];


const pipePromises = new Map();

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

export function euclideanDistance(A, B) {
  return Math.sqrt(A.reduce((sum, a, i) => sum + Math.pow(a - B[i], 2), 0));
}

export function sortArrayByKey(Arr, Key, ascending=true) {
  /*
  sortArrayByKey( [{'id':2},{'id':6},{'id':1},{'id':8},], 'id')
  
  */
  if(ascending == true){
  Arr.sort((a, b) => {
    if (a[Key] < b[Key]) return -1;
    if (a[Key] > b[Key]) return 1;
    return 0;
  });
}else{
  Arr.sort((a, b) => {
    if (a[Key] < b[Key]) return 1;
    if (a[Key] > b[Key]) return -1;
    return 0;
  });
  }
  return Arr
}