import { pipeline, AutoTokenizer } from "@huggingface/transformers";

let generator = null;
let tokenizer = null;

const model_id = 'onnx-community/SmolLM2-135M-Instruct'

self.onmessage = async (e) => {
    const { messages, model_id } = e.data;
    if(!generator){
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
    self.postMessage({status: 'complete', content: assistantReponse});
}