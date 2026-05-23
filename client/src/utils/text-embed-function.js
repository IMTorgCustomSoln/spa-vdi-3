//import { FeatureExtractionPipeline, pipeline, env } from "@huggingface/transformers";
import { toRaw } from "vue";
import { AutoModel, AutoTokenizer, env, Tensor } from "@huggingface/transformers";


//force the library to use the browser cache api instead of local asset files
env.allowLocalModels = false;
//critical: ensure local development ('/') and production assets match the same cache origin
env.useBrowserCacheURL = self?.location?.origin || '/';
//window?.location?.origin || 
env.useBrowserCache = true

class TextEmbeddingModel {

  constructor(){
    this.model_id = 'minishlab/potion-base-8M';
    this.model = null;
    this.tokenizer = null;
  }

  async initialize(){
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
    const embedding = Array.from( toRaw(embeddings)[0] )
    return embedding
  }
}


export const textEmbeddingModel = new TextEmbeddingModel()