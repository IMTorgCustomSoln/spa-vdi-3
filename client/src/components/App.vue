<template>

    <b-container id="app-content" fluid class="fluid-wide overflow-hidden">
        <NavbarTop @input="viewInput" />
        <b-container fluid class="fluid-wide">
            <div v-if="userContentStore.documentsIndex.documents.length > 0">
                <div v-show="appDisplayStore.views.viewSelection == 'search'">
                    <b-row>
                        <b-col>
                            <SearchBar :records="userContentStore.documentsIndex.documents"
                                v-on:search-table-results="searchTable">
                            </SearchBar>
                        </b-col>
                    </b-row>
                </div>
                <div>
                    <b-row>
                        <b-col cols="12">
                            <splitpanes class="default-theme" vertical style="height: 100%; width:100%;">
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

        Splitpanes, Pane
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
</style>