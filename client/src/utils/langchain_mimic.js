


/*
Key Logic Steps:
* Hierarchy: It prioritizes \n\n to keep paragraphs together, then \n for sentences, and finally " " for words.
* Recursion: If a string segment is still longer than chunkSize after splitting, it calls itself using the next available separator in the list.
* Merging: It combines smaller splits back together until they just barely fit under the chunkSize limit.
* Overlap: When moving to a new chunk, it "rewinds" and includes the last few segments from the previous chunk to maintain context.
Usage:
    const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 50, chunkOverlap: 10 });
    const chunks = splitter.splitText("Your very long string...");
*/
export class RecursiveCharacterTextSplitter {

  constructor({ chunkSize = 1000, chunkOverlap = 200, separators = ["\n\n", "\n", " ", ""] } = {}) {
    this.chunkSize = chunkSize;
    this.chunkOverlap = chunkOverlap;
    this.separators = separators;
  }

  splitText(text) {
    return this._recursiveSplit(text, this.separators);
  }

  _recursiveSplit(text, separators) {
    const finalChunks = [];
    let separator = separators[separators.length - 1];
    let nextSeparators = [];

    // Find the best separator that exists in the text
    for (let i = 0; i < separators.length; i++) {
      const s = separators[i];
      if (s === "") {
        separator = s;
        break;
      }
      if (text.includes(s)) {
        separator = s;
        nextSeparators = separators.slice(i + 1);
        break;
      }
    }

    const splits = separator !== "" ? text.split(separator) : text.split("");
    let goodSplits = [];

    for (const s of splits) {
      if (s.length < this.chunkSize) {
        goodSplits.push(s);
      } else {
        // If we have accumulated good splits, merge them before tackling the big one
        if (goodSplits.length > 0) {
          finalChunks.push(...this._mergeSplits(goodSplits, separator));
          goodSplits = [];
        }
        // Recursively split the part that is still too large
        finalChunks.push(...this._recursiveSplit(s, nextSeparators));
      }
    }

    if (goodSplits.length > 0) {
      finalChunks.push(...this._mergeSplits(goodSplits, separator));
    }

    return finalChunks;
  }

  _mergeSplits(splits, separator) {
    const docs = [];
    let currentDoc = [];
    let total = 0;

    for (const s of splits) {
      const len = s.length;
      const sepLen = currentDoc.length > 0 ? separator.length : 0;

      if (total + len + sepLen <= this.chunkSize) {
        currentDoc.push(s);
        total += len + sepLen;
      } else {
        if (currentDoc.length > 0) {
          docs.push(currentDoc.join(separator));
          
          // Rewind for overlap
          while (total > this.chunkOverlap || (total + len + sepLen > this.chunkSize && total > 0)) {
            const popped = currentDoc.shift();
            total -= popped.length + (currentDoc.length > 0 ? separator.length : 0);
          }
        }
        currentDoc.push(s);
        total += len + (currentDoc.length > 1 ? separator.length : 0);
      }
    }

    if (currentDoc.length > 0) {
      docs.push(currentDoc.join(separator));
    }
    return docs;
  }
}





import { pipeline } from "@huggingface/transformers";
import { summarizeText } from './summarize';

/** Transformers.js version of loadSummarizationChain

Usage:

const chain = await loadSummarizationChain({ type: "map_reduce" });
const result = await chain.call({
  input_documents: [
    "Long text chunk one describing the first half of a report...",
    "Long text chunk two describing the conclusion of the report..."
  ]
});
console.log("Summary:", result.text);

 */
export const loadSummarizationChain = async (options = { type: "stuff" }) => {
  const { type } = options;
  
  /* Initialize the local summarization pipeline
  const summarizer = await pipeline('summarization', 'Xenova/distilbart-cnn-6-6');

  const summarizeText = async (text) => {
    const output = await summarizer(text, {
      max_new_tokens: 100,
      chunk_length: 512, // Useful for models with strict token limits
    });
    // Transformers.js returns an array of objects: [{ summary_text: "..." }]
    return output[0].summary_text;
  };
  */

  const stuffChain = async (docs) => {
    const fullText = docs.join("\n\n");
    return await summarizeText(fullText);
  };

  const mapReduceChain = async (docs) => {
    // Map: Summarize each chunk individually
    const summaries = await Promise.all(docs.map(doc => summarizeText(doc)));
    // Reduce: Combine and summarize the summaries
    return await summarizeText(summaries.join("\n\n"));
  };

  return {
    call: async (input) => {
      const docs = input.input_documents;
      const text = type === "map_reduce" 
        ? await mapReduceChain(docs) 
        : await stuffChain(docs);
      return { text };
    },
  };
};






/**
 * Mimics langchain/core/language_models/llms BaseLLM and LLM classes
 */
class BaseLLM {
  constructor(fields = {}) {
    this.callbacks = fields.callbacks;
    this.tags = fields.tags || [];
    this.metadata = fields.metadata || {};
  }

  /**
   * The core entry point for users
   */
  async invoke(input, options = {}) {
    const result = await this.generate([input], options);
    // Returns the first generation text from the first prompt
    return result.generations[0][0].text;
  }

  /**
   * Handles batching and callback logic
   */
  async generate(prompts, options = {}) {
    const generations = await Promise.all(
      prompts.map((prompt) => this._generate(prompt, options))
    );
    
    return {
      generations, // Returns [[{text: '...'}]...]
      llmOutput: {} 
    };
  }

  /**
   * Internal method subclasses must implement
   */
  async _generate(prompt, options) {
    throw new Error("Method '_generate' must be implemented.");
  }

  _llmType() {
    throw new Error("Method '_llmType' must be implemented.");
  }
}




/**
 * Simplified LLM class that only requires implementing _call
 */
class LLM extends BaseLLM {
  /**
   * Implementation of _generate that wraps the simpler _call method
   */
  async _generate(prompt, options) {
    const text = await this._call(prompt, options);
    return [{ text }];
  }

  /**
   * Users override this method specifically
   */
  async _call(prompt, options) {
    throw new Error("Method '_call' must be implemented.");
  }
}





// --- Example Implementation: Local Transformers.js LLM ---

class TransformersLLM extends LLM {
  constructor(pipeline) {
    super();
    this.pipeline = pipeline;
  }

  _llmType() {
    return "transformers_js";
  }

  async _call(prompt, options) {
    const output = await this.pipeline(prompt, {
      max_new_tokens: 50,
      ...options
    });
    // Handle different pipeline output formats
    return Array.isArray(output) ? output[0].generated_text : output;
  }
}

/* --- Usage ---
const pipe = await pipeline('text-generation', 'Xenova/gpt2');
const myLlm = new TransformersLLM(pipe);
const response = await myLlm.invoke("The weather today is");
console.log(response);
*/




// Basic Message structure
class BaseMessage {
  constructor(content) {
    this.content = content;
  }
}

class SystemMessage extends BaseMessage {}
class HumanMessage extends BaseMessage {}
class AIMessage extends BaseMessage {}

/**
 * Mimics langchain/core/language_models/chat_models
 */
class BaseChatModel {
  constructor(fields = {}) {
    this.metadata = fields.metadata || {};
  }

  async invoke(messages, options = {}) {
    const result = await this._generate(messages, options);
    return result;
  }

  // Abstract: Subclasses implement this
  async _generate(messages, options) {
    throw new Error("Method '_generate' must be implemented.");
  }
}



/*
Features:
* Message Objects: Instead of raw strings, it accepts an array of message instances, 
allowing you to explicitly separate instructions (System) from user input (Human).
* Chat Templates: It uses the pipeline's built-in ability to handle chat-formatted 
arrays, which automatically adds the correct special tokens (like <|im_start|>) 
required by specific models.
* Response Wrapping: It returns an AIMessage object rather than just a string, 
preserving the LangChain structure.

*/
class TransformersChatModel extends BaseChatModel {
  constructor(pipeline) {
    super();
    this.pipeline = pipeline;
  }

  /**
   * Translates LangChain-style messages to a model-specific string
   */
  _formatMessages(messages) {
    return messages.map(m => {
      let role = 'user';
      if (m instanceof SystemMessage) role = 'system';
      if (m instanceof AIMessage) role = 'assistant';
      
      return { role, content: m.content };
    });
  }

  async _generate(messages, options) {
    const formattedMessages = this._formatMessages(messages);
    
    // Most modern Transformers.js text models support apply_chat_template automatically
    const output = await this.pipeline(formattedMessages, {
      max_new_tokens: 128,
      ...options
    });

    // Extract the generated text
    const text = output[0].generated_text.at(-1).content;
    return new AIMessage(text);
  }
}


/* --- Usage ---

async function runChat() {
  // 1. Load a text-generation pipeline
  const pipe = await pipeline('text-generation', 'Xenova/Qwen2.5-0.5B-Instruct');
  
  // 2. Initialize our custom chat model
  const chatModel = new TransformersChatModel(pipe);

  // 3. Define messages
  const messages = [
    new SystemMessage("You are a helpful assistant that answers in pirate speak."),
    new HumanMessage("What is the best way to bake a cake?")
  ];

  // 4. Invoke
  console.log("Generating response...");
  const response = await chatModel.invoke(messages);

  console.log("AI Response:", response.content);
}

runChat();


*/



/**
 * Base class for individual message templates
 */
class BaseMessagePromptTemplate {
  constructor(prompt) {
    this.prompt = prompt;
  }

  // Simple regex-based variable injection
  format(values) {
    const content = this.prompt.replace(/{(\w+)}/g, (match, key) => {
      return values[key] !== undefined ? values[key] : match;
    });
    return content;
  }
}

class SystemMessagePromptTemplate extends BaseMessagePromptTemplate {
  createMessage(values) { return new SystemMessage(this.format(values)); }
}

class HumanMessagePromptTemplate extends BaseMessagePromptTemplate {
  createMessage(values) { return new HumanMessage(this.format(values)); }
}

/**
 * Orchestrates multiple message templates
 */
class ChatPromptTemplate {
  constructor(promptMessages) {
    this.promptMessages = promptMessages;
  }

  static fromMessages(messages) {
    const templates = messages.map(([role, template]) => {
      if (role === "system") return new SystemMessagePromptTemplate(template);
      if (role === "human" || role === "user") return new HumanMessagePromptTemplate(template);
      throw new Error(`Role ${role} not supported`);
    });
    return new ChatPromptTemplate(templates);
  }

  async formatMessages(values) {
    return this.promptMessages.map(template => template.createMessage(values));
  }
}




/* --- Usage ---

async function runTemplatedChat() {
  // 1. Setup the Template
  const chatPrompt = ChatPromptTemplate.fromMessages([
    ["system", "You are a professional {job_title}. Answer questions in a {tone} tone."],
    ["human", "Tell me about {topic}."]
  ]);

  // 2. Inject Variables
  const formattedMessages = await chatPrompt.formatMessages({
    job_title: "Chef",
    tone: "passionate",
    topic: "the Maillard reaction"
  });

  // 3. Initialize Model (Using our previously defined TransformersChatModel)
  const pipe = await pipeline('text-generation', 'Xenova/Qwen2.5-0.5B-Instruct');
  const chatModel = new TransformersChatModel(pipe);

  // 4. Invoke
  const response = await chatModel.invoke(formattedMessages);
  
  // For demonstration, let's see the formatted output:
  console.log(formattedMessages);
}

runTemplatedChat();
/* 
Output:
[
  SystemMessage { content: 'You are a professional Chef. Answer questions in a passionate tone.' },
  HumanMessage { content: 'Tell me about the Maillard reaction.' }
]
*/