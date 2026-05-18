<template>
    <div class="explore-panel">
    <h6 class="panel-header">Document Explorer</h6>

    <div class="conversation-history">
        <div v-for="(msg, idx) in conversationHistory.slice(1)" :key="idx"
            :class="['message', msg.role]">
            <div class="message-role">{{  msg.role === 'user' ? 'You' : 'AI' }}</div>
            <div class="message-content">{{  msg.content }}</div>
        </div>

        <div v-if="isProcessing" class="message assistant processing">
            <div class="message-role">AI</div>
            <div class="message-content">{{  responseText || 'Thinking...' }}</div>
        </div>
    </div>
</div>
</template>


<script>
import { toRaw } from 'vue'
import { mapStores } from 'pinia'
import { useAppDisplay } from '@/stores/AppDisplay'
import { useUserContent } from '@/stores/UserContent'

import { getChatResponse } from '@/utils/chat-function'


export default{
    name:"ExploreResponse",
    props:{
        records: Array,
        search: Object,
        chatSubmit: Boolean,
        query: String
    },
    watch: {
        records: {
            handler: function (newVal, oldVal){
                if(Array.isArray(this.$props.records) && this.$props.records.length > 0){
                    console.log('Records updated')
                }
            }, 
            deep: false,

        },
        search: {
            handler: function (newVal, oldVal){
                if(typeof(this.$props.search) == 'object'){
                    //this.filterTable()
                    console.log('Search results updated:', this.$props.search)
                }
            }, 
            deep: false
            //immediate: true
        },
        chatSubmit:{
            handler: function (newVal, oldVal){
                if(newVal !== oldVal && this.$props.query){
                console.log('Chat submit triggered with query:', this.$props.query)
                this.getResponseText()
                }
            },
            deep: true,
            immediate: true
        },
    },
    data(){
        return {
            responseText: '',
            conversationHistory: [
                {role: 'system', content: 'You are a concise AI assistant helping analyze documents.'}
            ],
            worker: null,
            isProcessing: false
        }
    },
    computed:{
        ...mapStores(useAppDisplay, useUserContent),
        getDocument() {
            const docId = this.userContentStore.getSelectedDocument
            return this.userContentStore.documentsIndex.documents.filter(item => item.id==docId)[0]         //TODO:must use the Table array that is sorted on Score o/w incorrect
        },
    },
    methods:{
        async getResponseText(){
            console.log('=== ExploreResponse.getResponseText() called ===')
            console.log('Query:', this.$props.query)
            console.log('Search results:', this.$props.search)

            if (this.isProcessing){
                console.warn('Already processing a request, ignoring')
                return
            }
            this.isProcessing = true
            // 1. Get relevant document chunks from search results
            const contextChunks = await this.getRelevantContext()

            // 2. Build RAG prompt with context
            const systemPrompt = this.buildSystemPrompt(contextChunks)
            const userPrompt = this.$props.query
            console.log('System prompt length:', systemPrompt.length)
            console.log('Context chunks retrieved:', contextChunks.length)

            // 3. Update converation history
            this.conversationHistory[0].content = systemPrompt
            this.conversationHistory.push({ role: 'user', content: userPrompt})
            console.log('Convsation history:', this.conversationHistory)

            // 4. Send to worker
            if (!this.worker){
                this.worker = new Worker(new URL('@/utils/chat-worker.js', import.meta.url), {type: 'module'})
                this.worker.onmessage = (e) => this.handleWorkerResponse(e)
                this.worker.onerror = (error) => {
                    console.log('Worker error:', error.message)
                    this.isProcessing = false
                }
            }
            this.worker.postMessage({
                message: toRaw(this.conversationHistory)
            })
            console.log('AI is thinking...')
        },

        async getRelevantContext(){
            const contextChunks = []
            const resultGroups = this.$props.search.resultGroups || []

            //get top 5 documents by score
            const topDocs = resultGroups
                .filter(group => group.score > 0)
                .sort((a, b) => parseFloat(b.score) - parseFloat(a.score))
                .slice(0,5)
            
            //for each document, get teh matching text chcunks
            for (const group of topDocs){
                const doc = this.$props.records.find(r => r.id === group.ref)
                if (!doc) continue

                //use the phrase array which contains matching chunks from concept search
                const chunks = group.phrase || []
                for (const chunk of chunks.slice(0,3)){
                    contextChunks.push({
                        title: doc.title,
                        text: chunk,
                        score: group.score
                    })
                }
            }
            console.log(`Retrieved ${contextChunks.length} context chunks from ${topDocs.length} documents`)
            return contextChunks
        },

        buildSystemPrompt(contextChunks){
            if (contextChunks.length === 0){
                return 'You are a concise AI assistant. Answer the user\'s question directly.'
            }
            let prompt = 'You are a helpful AI assistant. Answer the user\'s question based on teh following document excerpts:\n\n'
            contextChunks.forEach((chunk, idx) => {
                prompt += `[Document ${idx + 1}: "${chunk.title}"]\n${chunk.text}\n\n`
            })
            prompt += 'Answer concisely baed on the provided context.  If the context doesn\'t contain relevant information, say so.'
            return prompt
        },

        handleWorkerResponse(e){
            const {status, content} = e.data
            console.log('Worker response:', e.data)
            if (status === 'update'){
                this.responseText = content
            } else if (status === 'complete') {
                console.log("Assistant response complete")
                this.responseText = content
                this.conversationHistory.push({ role: "assistant", content: content})
                this.isProcessing = false
            }
        }
    }
}
</script>

<style scoped>
.explore-panel {
    padding: 15px;
    height: calc(100vh - 150px);
    overflow-y: auto
}

.panel-header {
    text-align: center;
    margin-bottom: 20px;
    border-bottom: 2px solid #ddd;
    padding-bottom: 10px;
}

.conversation-history {
    display: flex;
    flex-direction: column;
    gap: 15px;
}

.message {
    padding: 10px 15px;
    border-radius: 8px;
    max-width: 85%;
}

.message.user {
    background-color: #e3f2fd;
    align-self: flex-end;
    margin-left: auto;
}

.message.assistant {
    background-color: #f5f5f5;
    align-self: flex-start;
}

.message.processing {
    opacity: 0.7;
    font-style: italic;
}

.message-role {
    font-weight: bold;
    font-size: 0.85em;
    margin-bottom: 5px;
    color: #666;
}

.message-content {
    white-space: pre-wrap;
    word-wrap: break-word;
}

</style>