<template>
    <h6 class="center">
        Response
    </h6>

<div class="block">
{{ responseText }}
</div>
</template>


<script>
import { toRaw } from 'vue'
import { mapStores } from 'pinia'
import { useAppDisplay } from '@/stores/AppDisplay'
import { useUserContent } from '@/stores/UserContent'

export default{
    name:"ExploreResponse",
    props:{
        records: Array,
        search: Object,
        chatSubmit: Boolean
    },
    watch: {
        records: {
            handler: function (newVal, oldVal){
                if(Array.isArray(this.$props.records) && this.$props.records.length > 0){
                    console.log('hi')
                }
            }, deep: false
        },
        search: {
            handler: function (newVal1, oldVal){
                if(typeof(this.$props.search) == 'object'){
                    //this.filterTable()
                }
            }, deep: false
        },
        chatSubmit:{
            handler: function (newVal, oldVal){
                console.log(`new chatSubmit value: ${newVal}`)
            },
            deep: false
        },
    },
    data(){
        return {
            responseText: null,
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
        getResponseText(){
            const worker = new Worker('@/utils/chat-worker.js', {type: 'module'});
            let conversationHistory = [
                { role: 'system', content: 'You are a concise local AI assistant.'}
            ];
            conversationHistory.push({ role: 'user', content: userText});
            worker.postMessage({
                messages: conversationHistory
            });
            console.log('AI is thinking...');
            worker.onmessage = (e) => {
                const {status, content }= e.data;
                if (status === 'complete'){
                    console.log("Assistant:", content);
                    conversationHistory.push({ role: "assistant", content: content});
                }
            };
            worker.onerror = function(error){
                console.error('Worker error:', error.message)
            };
        }
    }
}
</script>