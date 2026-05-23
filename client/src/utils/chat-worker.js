import { chatModel } from "./chat-function";

self.onmessage = async (e) => {
    try {
    //const { messages, model_id } = e.data;
    const assistantReponse = await chatModel.getChatResponse(e.data.message);
    self.postMessage({status: 'complete', content: assistantReponse});
    } catch (error) {
        self.postMessage({ status: 'error', content: null, error: error.message})
    }
}