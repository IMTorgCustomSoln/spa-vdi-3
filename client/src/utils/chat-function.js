import { pipeline, AutoTokenizer, env } from "@huggingface/transformers";
import { isModelCached } from "@/utils/utils";

/*
//define your conversation history (maintain context)
//maintain context by continuously appending both user inputs and assistant responses to the messages array.
let messages = [
    { role: "system", content: "You are a helpful coding assistant."},
    { role: "user", content: "How do I use local storage in JS?"},
];
*/

//force the library to use the browser cache api instead of local asset files
env.allowLocalModels = false;
//critical: ensure local development ('/') and production assets match the same cache origin
env.useBrowserCacheURL = self?.location?.origin || '/';
//window?.location?.origin || 

class ChatModel{

    constructor(){
        this.model_id = 'Xenova/Qwen1.5-0.5B-Chat';
        this.generator = null;
        this.tokenizer = null;
        this.defaultSystemPrompt = 'You are a concise AI assistant. Summarize the user\'s statement.'
    }

    async initialize(){
        //const model_id = 'Xenova/Qwen1.5-0.5B-Chat'//'onnx-community/SmolLM2-135M-Instruct'
        //let checkCached = isModelCached(model_id)
        env.useBrowserCache = true;
        if(!this.generator){
            this.generator = await pipeline('text-generation', this.model_id, {
                device: 'webgpu',
                quantized: true,
                dtype: 'q4'//'fp16',
            });
        }
        if(!this.tokenizer){
            this.tokenizer = await AutoTokenizer.from_pretrained(this.model_id);
        }
        return true
    }

    preparePrompt(text){
        const prompt = []
        prompt.push({role: 'system', content: this.defaultSystemPrompt})
        prompt.push({ role: 'user', content: text})
        return prompt
    }

    async getChatResponse(messages){
        if (!this.generator){
            await this.initialize()
        }
        let preparedMessages = null;
        if (typeof(messages)=='string'){
            preparedMessages = this.preparePrompt(messages)
        } else {
            preparedMessages = messages
        }
        const prompt = await this.tokenizer.apply_chat_template(preparedMessages, {
            tokenize: false,
            add_generator_prompt: true,
        });
        const output = await this.generator(prompt, {
            max_new_tokens: 256,
            temperature: 0.7,
            return_full_text: false,
            //callback to stream tokens to ui
            on_callback_function: (beams) => {
                const decoded = tokenizer.decode(beams[0].output_token_ids, {skip_special_tokens: true});
                self.postMessage({ status: 'update', content: decoded });
            }
        });
        const assistantReponse = output[0].generated_text    //.replace(prompt, '').trim();
        return assistantReponse;
        };
    }


export const chatModel = new ChatModel()