import { FeatureExtractionPipeline, pipeline, env } from "@huggingface/transformers";



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

export let extractor = null;
export async function getExtractor() {
    if (!extractor) {
        env.allowLocalModels = false;
        env.useBrowserCache = false;
        extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    }
    return extractor;
};

export async function getVectorFromText(text, modelName){
  env.allowLocalModels = true
  env.useBrowserCache = true
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