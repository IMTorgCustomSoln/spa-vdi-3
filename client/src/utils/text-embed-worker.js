import { textEmbeddingModel } from './text-embed-function';


self.onmessage = async (e) => {

  if('data' in e.data){
    try {
        const id = e.data.id
        const text = e.data.data.text
        // Run inference
        const output = await textEmbeddingModel.run(text)

        /* Send back the result (output.data is a Float32Array)
        self.postMessage({
            success: true,
            result: {
              page: data.page, 
              index: data.index,
              text: data.text,
              embedding: output.data
            },
            error: null
        })
        //}, [output.data.buffer]);
    } catch (error) {
        self.postMessage({
            success: false,
            error: error.message
        });
    */
      const vectorItem = e.data.data
      vectorItem['embedding'] = output
      self.postMessage({id, res: vectorItem});
    } catch (error) {
      self.postMessage({id, res: null, error: error.message });
    }

    }
};