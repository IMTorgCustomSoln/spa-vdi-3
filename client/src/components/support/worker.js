//import { DEFAULT_MODEL_NAME, getVectorFromText } from './vector.js';
//import * as vector  from './vector.js';
import { FeatureExtractionPipeline, pipeline } from "@xenova/transformers";
import { getFromMapOrCreate } from 'rxdb/plugins/core';



export const modelNames = [
  'Xenova/all-MiniLM-L6-v2',
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





onmessage = async (e) => {
    const embedding = await getVectorFromText(e.data.text, DEFAULT_MODEL_NAME);
    postMessage({
        id: e.data.id,
        embedding
    });
};