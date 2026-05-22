//import { FeatureExtractionPipeline, pipeline, env } from "@huggingface/transformers";
import { toRaw } from "vue";


/**
 * You can try different models:
 * @link https://huggingface.co/models?pipeline_tag=feature-extraction&library=transformers.js
 *
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
*/
import { AutoModel, AutoTokenizer, env, Tensor } from "@huggingface/transformers";


//force the library to use the browser cache api instead of local asset files
env.allowLocalModels = false;
//critical: ensure local development ('/') and production assets match the same cache origin
env.useBrowserCacheURL = self?.location?.origin || '/';
//window?.location?.origin || 

class TextEmbeddingModel {

  constructor(){
    this.model_id = 'minishlab/potion-base-8M';
    this.model = null;
    this.tokenizer = null;
  }

  async initialize(){
    //const use_cache = false;
    //env.allowLocalModels = use_cache;
    env.useBrowserCache = true
    if(!this.model){
      this.model = await AutoModel.from_pretrained(this.model_id, {
        config: {model_type: 'model2vec'},
        dtype: 'fp32'
      });
    }
    if(!this.tokenizer){
      this.tokenizer = await AutoTokenizer.from_pretrained(this.model_id);
      return true
    }
  }

  async run(text){
    const texts = [text];
    if(!this.model){ await this.initialize() }
    const { input_ids } = await this.tokenizer(texts, { 
      //device: hasWebGpu ? 'webgpu' : 'wasm',
      device: 'webgpu',
      add_special_tokens: false, 
      return_tensor: false 
    });

    const cumsum = arr => arr.reduce((acc, num, i) => [...acc, num + (acc[i - 1] || 0)], []);
    const offsets = [0, ...cumsum(input_ids.slice(0, -1).map(x => x.length))];

    const flattened_input_ids = input_ids.flat();
    const model_inputs = {
        input_ids: new Tensor('int64', flattened_input_ids, [flattened_input_ids.length]),
        offsets: new Tensor('int64', offsets, [offsets.length]),
    }
    const { embeddings } = await this.model(model_inputs);
    //console.log(embeddings.tolist());
    const embedding = Array.from( toRaw(embeddings)[0] )
    return embedding
  }
}


export const textEmbeddingModel = new TextEmbeddingModel()


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
};*/
/*
export async function getVectorFromText(text){
  env.allowLocalModels = false
  env.useBrowserCache = false
  const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2')    //TODO: https://jsfiddle.net/o35ryzfw/
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