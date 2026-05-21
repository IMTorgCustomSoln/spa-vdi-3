
import { FeatureExtractionPipeline, pipeline, env } from "@huggingface/transformers";
import { textEmbeddingModel } from "@/utils/text-embed-function"
import { chatModel } from "./chat-function";


export async function initializeModels(){
    const check_text_embedding = await textEmbeddingModel.initialize()
    const check_chat_model = await chatModel.initialize()
    const all_checks = [
        check_text_embedding,
        check_chat_model
    ].every(item => item === true)
    if( all_checks ){
        return true
    } else {
        return false
    }
}