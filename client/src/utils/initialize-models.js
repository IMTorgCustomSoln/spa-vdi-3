
import { FeatureExtractionPipeline, pipeline, env } from "@huggingface/transformers";



export async function initializeModels(){
    //TODO:add env var as args
    env.allowLocalModels = true;  //disable local model checks
    env.useBrowserCache = true;   //disable browser cache

    const hasWebGpu = !!navigator.gpu
    const summarizer = await pipeline(
            'summarization',
            'Xenova/distilbart-cnn-6-6',
            {
                device: hasWebGpu ? 'webgpu' : 'wasm',
                dtype: 'fp32', 
            })
    const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2')
    const testText = 'This is a test text.'
    const embedding_result = await extractor(testText, {pooling: "mean", normalize: true})
    const summary_result = await summarizer(testText, {max_new_tokens:100})
    if ( embedding_result != null & summary_result != null ){
        return true
    } else {
        return false
    }
}