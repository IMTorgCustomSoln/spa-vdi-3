<template>
    <!--
    <div style="height: 100%">
        <VList :data="getSnippets()" :style="{ height: '100%', padding: '10px' }" #default="item">
            <div v-html="item.indexed_snippet" class="snippet" @click="selectSnippetPage(item.idx, item.snippet)"
                style="padding-bottom: 15px;">
            </div>
        </VList>
    </div>-->


    <div v-if="getSnippets.length < 1" @click="selectSnippetPage(this.userContentStore.getSelectedDocument, '')">
        Select a document from the table or `PRESS` to view document pages.
    </div>
    <div v-else id="search-results" v-for="(snippet, index) in getSnippets">
        <div class="snippet" @click="selectSnippetPage(index, snippet)">
            <div v-html="snippet"></div>
            <!--
                                        <span :id="row.item.filepath + '-index_' + index" v-html="snippet"></span>
                                        <b-button size="sm" v-on:click="postNote($event)">Note
                                        </b-button>-->
        </div>
        <br />
    </div>
</template>

<script>
import { toRaw } from "vue";
//import { VList } from "virtua/vue";

import { mapStores } from 'pinia'
import { useUserContent } from '@/stores/UserContent'


export default {
    name: 'SnippetsScroll',
    components: {
        //VList
    },
    props: {
        snippets: Array
    },
    data() {
        return {}
    },
    computed: {
        ...mapStores(useUserContent),
    },
    methods: {
        addIndexToSnippetHtml(html, index) {
            return `${html.slice(0, 3)}${index + 1} - ${html.slice(3)}`
        },
        getSnippets() {
            const arr = toRaw(this.snippets)
            const results = []
            for (const [index, snippet] of arr.entries()) {
                const modified_snippet = this.addIndexToSnippetHtml(snippet, index)
                const record = {
                    idx: index,
                    snippet: snippet,
                    indexed_snippet: modified_snippet
                }
                results.push(record)
            }
            return results
        },
        selectSnippetPage(id, snippet) {
            //const mouseOverSnippet = `${id}-${snippet}`
            const tgtPage = parseInt(snippet.split('<b>pg.')[1].split('|')[0])
            const tgtText = snippet.split('<b style="background-color: yellow">')[1].split('</b>')[0]
            const mouseOverSnippet = { id: id, snippet: snippet, tgtPage: tgtPage, tgtText: tgtText }
            //this.searchResults = {...this.searchResults, mouseOverSnippet: mouseOverSnippet}
            this.mouseOverSnippet = mouseOverSnippet
            this.userContentStore.selectedSnippet = mouseOverSnippet
        },
    }
}
</script>

<style scoped>
#index {
    padding-right: 7px;
    font-weight: bold;
}

.snippet:hover {
    background-color: #ffeecf;
}

/*
#loader {
     
    display: flex;
    flex-direction: column;
    align-items: left;
    gap: 10px;
    padding: 20px;
    margin: 20px auto 20px auto;
    max-width: 400px;//

    max-height: calc(100vh - 50px - 200px);
    /*TODO:make more accurate
    overflow-y: scroll;
}*/

/*
.result {
    font-weight: 300;
    width: 80%;
    padding: 20px;
    text-align: center;
    background: #eceef0;
    border-radius: 10px;
}*/
</style>