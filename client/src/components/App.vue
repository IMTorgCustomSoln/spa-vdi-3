<template>

    <b-container id="app-content" fluid class="fluid-wide overflow-hidden">
        <NavbarTop @input="viewInput" />
        <b-container fluid class="fluid-wide">
            <div v-if="userContentStore.documentsIndex.documents.length > 0">
                <div v-show="['search', 'explore'].includes(appDisplayStore.views.viewSelection)">
                    <b-row>
                        <b-col>
                            <SearchBar :records="userContentStore.documentsIndex.documents"
                                v-on:search-table-results="searchTable"
                                v-on:chat-submit="handleChatSubmit">
                            </SearchBar>
                        </b-col>
                    </b-row>
                </div>
                <div>
                    <b-row>
                        <b-col cols="12">
                            <splitpanes class="default-theme" vertical style="height: 100%; width:100%;">
                                <!-- Search -->
                                <!--TODO: `height: calc(100vh - 130px)` works for 'Read' tab but not 'Search'-->
                                <pane :size="this.appDisplayStore.views.attrs.table.size">
                                    <Table :records="userContentStore.documentsIndex.documents"
                                        :search="searchTableResults"
                                        :tableFields="this.appDisplayStore.views.attrs.table.fields"
                                        :expansionBtn="this.appDisplayStore.views.attrs.table.toggleExpansionBtn">
                                        {{ createTable }}
                                    </Table>
                                </pane>

                                <pane :size="this.appDisplayStore.views.attrs.pdfViewer.size">
                                    <!-- Read -->
                                    <div
                                        v-if="appDisplayStore.views.viewSelection == 'read' && userContentStore.documentsIndex.documents.length > 0">

                                        <div class="viewer">
                                            <div v-if="appDisplayStore.pdfViewerAvailable">
                                                <PdfViewer />
                                            </div>
                                            <div v-else>
                                                <PdfPlaceholder />
                                            </div>
                                        </div>
                                    </div>

                                    <!-- Explore -->
                                    <div
                                        v-else-if="appDisplayStore.views.viewSelection == 'explore' && userContentStore.documentsIndex.documents.length > 0">

                                        <div class="explore">
                                            <div>
                                                <ExploreResponse  :records="userContentStore.documentsIndex.documents"
                                                    :search="searchTableResults"
                                                    :chatSubmit="appDisplayStore.aiConfigs.chatSubmitBtn"
                                                    :query="searchTableResults.query">
                                                    >
                                                </ExploreResponse>
                                            </div>
                                        </div>
                                    </div>


                                </pane>
                            </splitpanes>
                        </b-col>
                    </b-row>
                </div>
            </div>
        </b-container>
    </b-container>
</template>


<script>
import { Splitpanes, Pane } from 'splitpanes'
import 'splitpanes/dist/splitpanes.css'

import NavbarTop from '@/components/NavbarTop.vue'
import SearchBar from '@/components/SearchBar.vue'
import Table from '@/components/Table.vue'
import PdfViewer from '@/components/PdfViewer.vue'
import PdfPlaceholder from '@/components/PdfPlaceholder.vue'
import ExploreResponse from '@/components/ExploreResponse.vue'

import { initializeModels } from '../utils/initialize-models'

import { mapStores } from 'pinia'
import { useAppDisplay } from '@/stores/AppDisplay'
import { useUserContent } from '@/stores/UserContent'


export default {
    name: 'App',
    components: {
        NavbarTop,
        SearchBar,
        Table,
        PdfViewer,
        PdfPlaceholder,
        Splitpanes, Pane,
        ExploreResponse
    },
    data() {
        return {
            view: {
                tableAttrs: {
                    colsTable: 12,
                    fields: [],
                    toggleExpansionBtn: true,
                },
                viewerAttrs: {
                    colsPdfViewer: 10,
                }
            },
            searchTableResults: {
                query: '',
                searchTerms: [],
                resultIds: [],
                resultGroups: []
            },
            pdfViewerAvailable: true     // => appDisplayStore.pdfViewerAvailable
        }
    },
    async mounted(){
        const result = true //await initializeModels()
        if (result){
            console.log('AI models initialized (if needed)')
        }
    },
    computed: {
        ...mapStores(useAppDisplay, useUserContent),
    },
    methods: {
        searchTable(results) {
            this.searchTableResults = { ...this.searchTableResults, query: results.query }
            this.searchTableResults = { ...this.searchTableResults, searchTerms: results.searchTerms }
            this.searchTableResults = { ...this.searchTableResults, resultIds: results.resultIds }
            this.searchTableResults = { ...this.searchTableResults, resultGroups: results.resultGroups }
        },
        handleChatSubmit(){
            this.appDisplayStore.aiConfigs.chatSubmitBtn = !this.appDisplayStore.aiConfigs.chatSubmitBtn
        },
        toggleChatSidebar() {
            this.isSidebarOpen = !this.isSidebarOpen;
            // CRITICAL: Force the browser to evaluate layout sizes 
            // This instantly kicks off your `handleResize` function inside PdfViewer.vue
            this.$nextTick(() => {
              window.dispatchEvent(new Event('resize'));
            });
        },
    }
}

</script>

<style scoped>
#app-content {
    padding: 0px;
    margin: 0px;
    height: 100vh;
}

.navbar {
    padding: 0;
}

/*
.fluid-wide {
    max-width: 2200px;
}*/

.viewer {
    margin-left: 5px;
    margin-right: 5px;
}

.app-container {
  display: grid;
  grid-template-columns: minmax(0, 1fr) var(--sidebar-width, 400px); /* Strict boundary locking */
  height: 100vh;
  width: 100vw;
  overflow: hidden;
}

.pdf-viewer-panel {
  min-width: 0; /* CRITICAL Vue 3 layout fix: allows child canvas wrappers to shrink safely */
  overflow: auto;
  position: relative;
}
</style>