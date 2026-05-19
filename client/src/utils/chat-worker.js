import { getChatResponse } from "./chat-function";

self.onmessage = async (e) => {
    //const { messages, model_id } = e.data;
    const assistantReponse = await getChatResponse(e.data.message);
    self.postMessage({status: 'complete', content: assistantReponse});
}