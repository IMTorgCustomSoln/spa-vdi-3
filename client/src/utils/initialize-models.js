/*
Notes:
* this should work because the models are caching files in the same place by
applying the correct hf global variables.
* also, by working with the models, directly, instead of using Workers, the
app does not have the overhead of building and terminating workers
*/
import { FeatureExtractionPipeline, pipeline, env } from "@huggingface/transformers";
import { textEmbeddingModel } from "@/utils/text-embed-function"
import { chatModel } from "./chat-function";
import { summaryModel } from "./summarize-function";


export async function initializeModels(){
    const check_text_embedding = await textEmbeddingModel.initialize()
    const check_chat_model = await chatModel.initialize()
    const check_summary_model = await summaryModel.initialize()
    const all_checks = [
        check_text_embedding,
        check_chat_model,
        check_summary_model
    ].every(item => item === true)
    if( all_checks ){
        return true
    } else {
        return false
    }
}