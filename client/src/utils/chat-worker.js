import { getChatResponse } from "./chat-function";

self.onmessage = async (e) => {
    const { messages, model_id } = e.data;
    const assistantReponse = await getChatResponse(messages, model_id);
    self.postMessage({status: 'complete', content: assistantReponse});
}