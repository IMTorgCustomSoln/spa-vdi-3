<template>
    <div v-if="expansionBtn">
        <b-button size="sm" variant="primary" v-on:click="expandAll" class="fixed-medium">Expand All</b-button>
        <b-button size="sm" variant="primary" v-on:click="collapseAll" class="fixed-medium">Collapse All</b-button>
    </div>
    <div v-if="this.userContentStore.documentsIndex.documents">
        <!--refs
            * showDetails: https://stackoverflow.com/questions/52327549/bootstrap-vue-table-show-details-when-row-clicked
            * reactivity: https://github.com/bootstrap-vue/bootstrap-vue/issues/2960
            * max recursion error can occur if filtering or other props are not correct
        -->
        <b-row>
            <b-col :cols="this.appDisplayStore.views.attrs.table.colsTable">
                <b-table hover :items="items" :fields="fields" :filter="tableFilter" :filter-function="onFiltered"
                    :sort-by.sync="sortBy" :sort-desc.sync="sortDesc" :sort-direction="desc" :head-variant="headVariant"
                    primary-key='id' striped small responsive="sm" sticky-header="1000px" bordered
                    thead-class="tableHead bg-dark text-white" select-mode="single" selectable
                    @row-clicked="expandAdditionalInfo">

                    <!-- Custom formatted header cell -->
                    <template #head(sort_key)="data">
                        <span class="col-info">{{ data.label.toUpperCase() }}
                            <Guide v-bind="guides.score" />
                        </span>
                    </template>
                    <template #head(hit_count)="data">
                        <span class="col-info">{{ data.label.toUpperCase() }}
                            <Guide v-bind="guides.hit" />
                        </span>
                    </template>

                    <template #cell(show_details)="row">
                        <!-- As `row.showDetails` is one-way, we call the toggleDetails function on @change -->
                        <b-form-checkbox v-model="row.detailsShowing" @change="row.toggleDetails">
                            -
                        </b-form-checkbox>
                    </template>

                    <template #row-details="row">
                        <b-card>
                            <b-row class="mb-2">
                                <b-col sm="6" class="text-sm-left">
                                    <b-card>
                                        <b-row>
                                            <b-col sm="5" class="text-sm-left">
                                                <b-row><b>Author: &nbsp</b> {{ row.item.author }}</b-row>
                                                <b-row><b>Subject: &nbsp</b> {{ row.item.subject }}</b-row>
                                                <b-row><b>Keywords: &nbsp</b> {{ row.item.keywords }}</b-row>
                                            </b-col>
                                            <b-col sm="7" class="text-sm-left">
                                                <b>Contents:</b> <br><span v-html="row.item.pp_toc.join('<br>')"></span>
                                            </b-col>
                                        </b-row>
                                    </b-card>
                                </b-col>

                                <b-col sm="6" class="text-sm-left">
                                    <div><b>Document summary: </b> <br />
                                        {{ row.item.summary }}
                                    </div>
                                    <!--
                    <div v-if="!totalDocuments"><b>Document summary: </b> <br/>
                        {{ row.item.summary }}
                    </div>
                    <div v-else><b>Search results in {{ row.item.hit_count }} hits: </b> <Guide v-bind="guides.snippet" /> </div>
                    <br/>
                    <div class="left_contentlist">
                        <div v-if="(row.item.hit_count > 0) && (row.item.snippets.length == 0)">
                            <span class="warningMsg">To view search result snippets, press `Collapse All`, then individually click the row(s).</span>
                        </div>
                        <div v-else class="itemconfiguration">
                            <div id="search-results" v-for="(snippet, index) in row.item.snippets">
                                <div class="snippet" v-on:mouseover="selectSnippetPage(index, snippet)">
                                    <span :id="row.item.filepath + '-index_' + index" v-html="snippet"></span>
                                    <b-button size="sm" v-on:click="postNote($event)">Note
                                    </b-button>
                                </div>
                                <br/>
                            </div>
                        </div>
                    </div> 
                    -->
                                </b-col>
                            </b-row>
                        </b-card>
                    </template>

                </b-table>
            </b-col>


            <b-col><!--:cols="this.appDisplayStore.views.attrs.table.collsSnippets">-->
                <!-- TODO: 
                    * create Search Snippets component
                    * keep guide with snippets
                    * create new guide for table: Score, Hits, ...
                
                -->
                <div
                    v-if="appDisplayStore.views.viewSelection == 'read' && userContentStore.documentsIndex.documents.length > 0">
                    <div class="itemconfiguration snippet_container">
                        <div style="color: black; text-align: center;">
                            <b>{{ this.getDocument.title }}</b>
                        </div>
                        <!--<h3>Search Results</h3>-->
                        <div class="snippet-header">
                            <b>Search results in {{ getSearchSnippetsLength }} text-block hits: </b>
                            <Guide v-bind="guides.snippet" />
                        </div>
                        <div v-if="getSearchSnippetsLength < 1"
                            @click="selectSnippetPage(this.userContentStore.getSelectedDocument, '')">
                            Select a document from the table or `PRESS` to display document pages.
                        </div>
                        <div v-else id="search-results">
                            <SnippetsScroll :snippets="getSearchSnippets" />
                        </div>
                    </div>
                </div>
            </b-col>

        </b-row>
    </div>
    <br>
    <br>
</template>


<script>
import { getDateFromJsNumber, getFormattedFileSize } from '@/components/support/utils.js'
import Guide from '@/components/support/Guide.vue'
import SnippetsScroll from '@/components/support/SnippetsInfinite.vue'

import { mapStores } from 'pinia'
import { useAppDisplay } from '@/stores/AppDisplay'
import { useUserContent } from '@/stores/UserContent'


export default {
    name: 'Table',
    props: {
        records: Array,
        search: Object,
        tableFields: Array,
        expansionBtn: Boolean
    },
    watch: {
        records: {
            handler: function (newVal, oldVal) {
                if (Array.isArray(this.$props.records) &&
                    this.$props.records.length > 0
                ) {
                    this.createTable()
                }
            },
            deep: false
        },
        search: {
            handler: function (newVal, oldVal) {
                if (typeof (this.$props.search) == 'object') {
                    this.filterTable()
                }
                //this.createTable()  => using this destroys changes to item fields, such as Score, Hit
            },
            deep: false
        },
        tableFields: {
            handler: function (newVal, oldVal) {
                if (typeof (this.$props.tableFields) == 'object') {
                    this.fields = this.$props.tableFields
                }
                //this.createTable()  => using this destroys changes to item fields, such as Score, Hit
            },
            deep: false
        },
        expansionBtn: {
            handler: function (newVal, oldVal) {
                this.showExpandBtn = newVal
            }
        }
    },
    //emits: ['send-note'],
    components: {
        Guide,
        SnippetsScroll
    },
    data() {
        return {
            fields: null,

            initializeTable: false,
            items: [],
            snippets: [],
            selectedItem: null,

            tableFilter: [],
            sortBy: 'sort_key',
            sortDesc: true,
            headVariant: 'dark',

            totalDocuments: 0,
            activeDetailsTab: 1,
            mouseOverSnippet: '',
            //displayLimit: 0,
            guides: {
                snippet: {
                    id: 'snippet',
                    title: 'Search Results',
                    markdown: `The search results display as snippets of text 
containing the highlighted search terms.  The begining of the text includes
the page number, and the location of text in characters from the begining of the
page, such as \`pg.3 | char.5340)\`.  

When the cursor passes over an individual result snippet, an orange background will
note its selection, and the document images (to the left) will display the page of 
the text.

At the end of the snippet of text is a \`Note\` button.  When clicked, the Managed
Notes sidebar displays and the text snippet appears in the Staging Area.  It is 
ready to be organized with the note Topics.`
                },
                score: {
                    id: 'score',
                    variantCustom: '#a4ff92',
                    title: '\'Text Classification\' Score Description',
                    markdown: `Score of how similar some sentences within the document 
are to pre-determined target phrases.  The higher the score, the more similar the text 
to what you're targeting.  This is inferred using custom-trained text classification 
models.  

_Note: models may classify text as being very similar to the targeted text; however, 
\`Hit\` count may be zero because it looks for specific terms._`
                },
                hit: {
                    id: 'hit',
                    variantCustom: '#a4ff92',
                    title: '\'Key Term\' Hit Count Description',
                    markdown: `Count of exact key terms found within a document.

_Note: \`Hit\` count may be zero because exact terms are not found, but \`Score\` can be
high because terms are found that are very similar to the exact terms._`
                },
            }
        }
    },
    mounted() {
        this.createTable()

    },
    computed: {
        ...mapStores(useUserContent, useAppDisplay),
        getSearchSnippets() {
            const selected = this.userContentStore.getSelectedDocument.toString()
            return this.items.filter(item => item.id == selected)[0].snippets
        },
        getSearchSnippetsLength() {
            return this.getSearchSnippets.length
        },
        getDocument() {
            const docId = this.userContentStore.getSelectedDocument
            return this.userContentStore.documentsIndex.documents.filter(item => item.id == docId)[0]         //TODO:must use the Table array that is sorted on Score o/w incorrect
        }
    },
    methods: {

        // Creation and search
        createTable() {
            // Populate the table with the transformed data records
            this.items.length = 0
            this.fields = this.$props.tableFields
            for (const record of this.$props.records) {
                const item = JSON.parse(JSON.stringify(record))
                this.items.push(item)
            }
            this.initializeTable = true
        },

        filterTable() {
            //filter table based on selected items
            //also include score, sort and row details' text
            //remove snippets display to improve performance
            this.collapseAll()
            this.totalDocuments = this.$props.search.resultIds.length
            this.tableFilter.length = 0
            if (this.$props.search.resultIds.length == 0) {
                this.resetAllItems()
            } else {
                this.items.map(item => {
                    if (this.$props.search.resultIds.includes(item.id)) {
                        //filter and sort table items 
                        this.tableFilter.push(item.id)
                        const idx = this.$props.search.resultGroups.map(resultFile => resultFile.ref).indexOf(item.id)
                        let resultFile = this.$props.search.resultGroups[idx]
                        if (idx <= -1) {
                            //empty search
                            this.resetItem(item)
                        } else if (resultFile.score == 'NaN') {
                            //this.searchModel()
                            const record = this.$props.records.filter(rec => rec.id == item.id)[0]
                            item.sort_key = record.sort_key
                            item.hit_count = record.hit_count
                            item.snippets.length = 0
                        } else {
                            //typical
                            item.sort_key = resultFile.score
                            item.hit_count = resultFile.count
                            item.snippets.length = 0
                        }
                    }
                })
                this.sortDesc = true
                this.activeDetailsTab = 1
                return true
            }
        },

        createSearchSnippets(row, MARGIN = 250, MAX_SUBGROUP = 10) {
            this.items.map(item => {
                if (row.id == item.id) {
                    const idx = this.$props.search.resultGroups.map(resultFile => resultFile.ref).indexOf(item.id)
                    if (idx >= 0) {
                        let resultFile = this.$props.search.resultGroups[idx]

                        //reset snippets
                        item.snippets.length = 0

                        //combine hits within the MARGIN space into same snippet
                        const positions = resultFile.positions.map(item => item)
                        const positionGroups = []
                        let incr = 0
                        for (let index = 0; (index + incr) < positions.length; index++) {
                            let indexCorrected = index + incr
                            const pos = positions[indexCorrected]
                            const subgroup = []
                            subgroup.push(pos)
                            if (indexCorrected + 1 == positions.length) {
                                positionGroups.push(subgroup)
                                break
                            } else {
                                for (let nextIndex = indexCorrected + 1; nextIndex < positions.length; nextIndex++) {
                                    const nextPos = positions[nextIndex]
                                    const diff = nextPos[0] - pos[0]
                                    if (diff < MARGIN * 2 & subgroup.length < MAX_SUBGROUP) {
                                        subgroup.push(nextPos)
                                        incr++
                                        if (index + incr + 1 == positions.length) {
                                            positionGroups.push(subgroup)
                                        }
                                    } else {
                                        positionGroups.push(subgroup)
                                        break
                                    }
                                }
                            }
                        }
                        console.log(`positionGroups (array (snippets) of arrays (hits)) for file id: ${item.id}`)
                        console.log(positionGroups)

                        //create array of snippts
                        if (positionGroups.length == 0) {
                            //item.snippets.push(null)          //TODO:this effects whether a snippet is displayed in Table!!!
                        } else {
                            for (let grp of positionGroups) {
                                const snippet = []
                                /*
                                item.body_chars - object of each page's length indexed from pageNum (unordered)
                                item.accumPageChars - array of each page's length indexed from first (zero-indexed) page (ordered)
    
                                Document Snippets
                                * resultGrps - array of all hits within a doc
                                * positionGroups - array (snippets) of arrays (hits)
                                * grp - snippet of targets
    
                                Starting Header references the position from which the snippet starts
                                * pageNum - page (human, 1-indexed) the snippet begins
                                * startFromPage - starting character for snippet on that page
                                * endPage - count of characters for that page
                                */
                                for (let [index, pos] of grp.entries()) {
                                    //initial target
                                    if (index == 0) {
                                        const start = pos[0] - MARGIN > 0 ? pos[0] - MARGIN : 0
                                        const pageIdx = item.accumPageChars.map(val => { return start < val }).indexOf(true)
                                        //starting header
                                        const pageNum = parseInt(pageIdx) + 1
                                        const startFromPage = pageIdx == 0 ? start : start - item.accumPageChars[pageIdx - 1]
                                        const endPage = item.body_chars[pageNum]
                                        //target
                                        const hightlight = item.html_body.slice(pos[0], pos[0] + pos[1])
                                        const hdr = `<b>pg.${pageNum.toString()}| char.${startFromPage}/${endPage})</b>  `
                                        const startText = item.html_body.slice(start, pos[0])
                                        const middleText = `<b style="background-color: yellow">${hightlight}</b>`
                                        //const startText = `<b>pg.${pageNum.toString()}| char.${startFromPage}/${endPage})</b>  ${item.html_body.slice(start, pos[0])}<b style="background-color: yellow">${hightlight}</b>`
                                        const endText = grp.length == 1 ? item.html_body.slice(pos[0] + pos[1], pos[0] + pos[1] + MARGIN) : ''
                                        //const text = startText + endText
                                        const text = hdr + startText + middleText + endText
                                        if (endPage < startFromPage) {
                                            console.log('stopped')
                                        }
                                        snippet.push(text)
                                        //middle targets
                                    } else if (index == grp.length - 1) {
                                        const middleStart = item.html_body.slice(grp[index - 1][0] + grp[index - 1][1], pos[0])
                                        const hightlight = item.html_body.slice(pos[0], pos[0] + pos[1])
                                        const end = pos[0] + pos[1] + MARGIN < item.html_body.length ? pos[0] + pos[1] + MARGIN : item.html_body.length
                                        const text = `${middleStart} <b style="background-color: yellow">${hightlight}</b> ${item.html_body.slice(pos[0] + pos[1], end)}`
                                        snippet.push(text)
                                        //end targets
                                    } else {
                                        const middleStart = item.html_body.slice(grp[index - 1][0] + grp[index - 1][1], pos[0])
                                        const hightlight = item.html_body.slice(pos[0], pos[0] + pos[1])
                                        const text = `${middleStart} <b style="background-color: yellow">${hightlight}</b>`
                                        snippet.push(text)
                                    }
                                }
                                item.snippets.push(snippet.join(''))
                            }
                        }
                        //END

                    }
                }
            })

        },

        onFiltered(row, filter) {
            // Applied to each table row to determine if it 
            //should be displayed   //TODO: does this need to be a computed instead of method?
            if (filter.length == 0) {
                return true;
            } else if (filter.includes(row.id)) {
                return true;
            } else {
                return false;
            }
        },

        resetItem(item) {
            item.sort_key = 0
            item.hit_count = 0
            item.snippets = []
        },

        resetAllItems() {
            this.items.map(item => {
                this.resetItem(item)
            })
            this.sortDesc = false
            this.tableFilter.length = 0
        },


        // Buttons and formatting
        expandAll() {
            this.items.map(item => this.$set(item, '_showDetails', true))
        },
        collapseAll() {
            this.items.map(item => this.$set(item, '_showDetails', false))
        },

        //TODO: despite the two below rows, changing the active tab to '1' (image) does not work
        expandAdditionalInfo(row) {
            //TODO:note
            this.selectedItem = row.id
            this.userContentStore.selectedDocument = row.id

            if (this.appDisplayStore.views.viewSelection == 'search') {
                row._showDetails = !row._showDetails
                row._activeDetailsTab = 1  //this.activeDetailsTab
                if (row._showDetails) {
                    this.createSearchSnippets(row)
                }
            } else if (this.appDisplayStore.views.viewSelection == 'read') {
                this.createSearchSnippets(row)
                //this.snippets = this.items.filter(item => item.id==this.selectedItem)[0].map(item => item.snippet)
            }
        },
        onTabChanged() {
            this.items.map(item => this.$set(item, '_activeDetailsTab', this.activeDetailsTab))
        },
        formatDateAssigned(value) {
            let dt = null
            if (Number.isInteger(value)) {
                //???
                dt = getDateFromJsNumber(value)
            } else if (typeof value === 'string' || value instanceof String) {
                //iso format from python
                dt = value.split('T')[0]
            }
            return dt;
        },
        getFormattedFileSize(value) {
            return getFormattedFileSize(value, 'unit')
        },
        getFormattedScore(value) {
            const decimal = Math.round(parseFloat(value) * 1000) / 1000
            return decimal
        },
        getFormattedPath(path) {
            return path ? path : './'
        },
        
        // Row details 
        selectSnippetPage(id, snippet) {
            //const mouseOverSnippet = `${id}-${snippet}`
            const mouseOverSnippet = { id: id, snippet: snippet }
            //this.searchResults = {...this.searchResults, mouseOverSnippet: mouseOverSnippet}
            this.mouseOverSnippet = mouseOverSnippet
            this.userContentStore.selectedSnippet = mouseOverSnippet
        },/* NOTE:this was used for an earlier button on the snippets
        postNote(event) {
            const element = event.target.parentElement.children[0]
            //TODO: fix the code below which should use `new NoteRecord()`, but from within Draggable - not here
            const noteItem = {
                id: element.id.toString(),
                list: 'stagingNotes',
                type: 'auto',
                innerHTML: element.innerHTML.toString(),
                innerText: element.innerText.toString()
            }
            //console.log(noteItem)

            //TODO:note
            this.$emit('send-note', noteItem);
        },*/
    }
}
</script>


<style scoped>
.col-info {
    color: #a4ff92;
}


.fixed-medium {
    width: 94px !important;
    margin: 5px;
}

#table-panel input {
    margin: 5px;
}

#table-panel button {
    margin: 5px;
}

#search-results {
    font-size: 12px;
    height: calc(100vh - 50px - 200px)
}

.errorMsg {
    color: red;
}

.warningMsg {
    color: orange;
}

.snippet>.btn-sm {
    font-size: 8px;
    padding: 2px;
    margin-left: 10px;
}

/*ref: http://jsfiddle.net/7w8TC/1/ */
.itemconfiguration {
    /*height:700px;       TODO: align height of snippets and image */
    width: 550px;
    overflow-y: auto;
    float: left;
    position: relative;
    margin-left: 5px;
}

.left_contentlist {
    width: 550px;
    float: left;
    padding: 0 0 0 5px;
    position: relative;
    float: left;
    border-right: 1px #f8f7f3 solid;
}



/*TODO:remove
.snippet:hover {
    background: #ffeecf;
}*/

.snippet-header {
    padding-bottom: 15px;
}

/*
.snippet {
    padding-bottom: 10px;
}


.snippet_container {
    height: 90vh;
    overflow-y: auto;
}*/
</style>