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
        tableFields: Array,
        expansionBtn: Boolean
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
        //getResponseText(){},
    },
    methods:{}
}
</script>