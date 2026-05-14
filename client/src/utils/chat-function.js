import { pipeline, AutoTokenizer, env } from "@huggingface/transformers";
import { isModelCached } from "./utils";

const model_id = 'onnx-community/SmolLM2-135M-Instruct'
let generator = null;
let tokenizer = null;


/*
//define your conversation history (maintain context)
//maintain context by continuously appending both user inputs and assistant responses to the messages array.
let messages = [
    { role: "system", content: "You are a helpful coding assistant."},
    { role: "user", content: "How do I use local storage in JS?"},
];
*/

async function getChatResponse(messages, model_id){

    let checkCached = isModelCached(model_id)

    if(!generator){
        env.allowLocalModels = false;
        env.useBrowserCache = false;
        generator = await pipeline('text-generation', model_id, {
            device: 'webgpu',
            dtype: 'fp16',
        });
        tokenizer = await AutoTokenizer.from_pretrained(model_id);
    }
    const prompt = tokenizer.apply_chat_template(messages, {
        tokenize: false,
        add_generator_prompt: true,
    });
    const output = await generator(prompt, {
        max_new_tokens: 256,
        temperature: 0.7,
        //callback to stream tokens to ui
        on_callback_function: (beams) => {
            const decoded = tokenizer.decode(beams[0].output_token_ids, {skip_special_tokens: true});
            self.postMessage({ status: 'update', content: decoded });
        }
    });
    const assistantReponse = output.generated_text.replace(prompt, '').trim();
    return assistantReponse;
};