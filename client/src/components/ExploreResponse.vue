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
    watch: {
        /*TODO, note: event source is Table Snippets
        'userContentStore.selectedSnippet': {
            handler: async function (newVal, oldVal) {
                let pg = 0
                let tgtText = ''
                if(newVal.snippet!=''){
                    const txtPg = parseInt(newVal.snippet.split('<b>pg.')[1].split('|')[0])
                    pg = txtPg <= 1 ? txtPg : txtPg - 1
                    tgtText = newVal.snippet.split('<b style="background-color: yellow">')[1].split('</b>')[0]
                }
                //const app = await this.getApp
                await this.loadDoc()
                //this.search(tgtText)
                //app.page = pg
            }
        }*/
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
        }
    },
    methods:{}
}
</script>