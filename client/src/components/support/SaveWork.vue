<template>
    <!-- Open modal button 
    TODO:disabled - determine how to save to file then update with each change 
    ref: https://github.com/IMTorgCustomSoln/visual-document-index/blob/master/components/support/SaveWork.vue
-->
    <b-button id='btnSaveContinue' v-b-modal="'save-continue-modal'" variant="primary" class="" :disabled="false">
        {{ btnText }}
    </b-button>

    <!-- modal -->
    <b-modal id='save-continue-modal' ok-only>
        <template #modal-title>
            Save current work session
        </template>
        <br>
        <!-- Explanation TODO:fix-->
        <div v-if="description">

            <p>
                <bold style="font-weight: bold">Without a save file configuration, closing your browser will cause all
                    work to
                    be lost.</bold> <br><br>
                You must manually save your Workspace to a file if you wish to return to it, in the future. This can be
                imported, later, to continue where you last saved.<br><br>

                <!--TODO:auto-save feature
            For the Workspace to be automatically saved you must configure a save file.  <bold style="font-weight: bold">Without a save file configuration, closing 
            your browser will cause all work to be lost.</bold> This can be imported, later, to continue where you last saved.<br><br> 
            -->
                <!--
            The current configuration is:
            <ul class="no-li-dot">
                <li><label for="fileName">File: &nbsp</label><output id="fileName">{{ config.fileName }}</output></li>
                <li><div>

                    
                    <label for="autoSave">Autosave: &nbsp</label>
                    <input 
                        type="checkbox"
                        v-model="config.autoSave"
                        value="true"
                        unchecked-value="false"
                        variant="primary"
                        >
                    

                </div>
                </li>
            </ul>-->


            <div v-if="resultDisplay.error" style="color: red">
                {{ resultDisplay.error }}
            </div>

            <em>Note: the saved file is typically quite large in size (several megabytes). If you want a light-weight
                file with
                only your managed notes (such as to share with team
                members), instead, open the <code>Notes Manager</code> sidebar and click <code>Export</code> > <code>Data
            Storage</code></em>.
            </p>
        </div>

        <!-- Control -->
        <template #modal-footer>
            <div v-if="description">
                <b-button @click="prepareWorkspace" class="btn-sm m-1" variant="primary">Prepare Workspace</b-button>
                <b-button @click="saveWorkStream" v-b-modal.modal-close_visit class="btn-sm m-1" variant="primary"
                    :disabled="config.disabled">Save Workspace</b-button>
            </div>
        </template>
    </b-modal>
</template>

<script>
import { isProxy, toRaw } from 'vue'

//import { DocumentIndexData, ManagedNotesData } from './data'
import { ExportAppStateFileName } from '@/stores/constants.js'
import { mapStores } from 'pinia'
import { useAppDisplay } from '@/stores/AppDisplay'
import { useUserContent } from '@/stores/UserContent'


export default {
    name: 'SaveWork',
    data() {
        return {
            btnText: 'Save',
            description: true,
            modified_docs: [],
            config: {
                disabled: true,
                fileHandle: '',
                fileName: '<TBD>',
                fileSize: '<TBD>',
                autoSave: true,
            },
            resultDisplay: {
                error: ''
            },
            //documentsIndex: DocumentIndexData,
            //managedNotes: ManagedNotesData,
        }
    },
    mounted() {
        this.$root.$on('bv::modal::hidden', (bvEvent, modalId) => {
            this.initializeFormValues()
        })
    },
    computed: {
        ...mapStores(useAppDisplay, useUserContent),

    },
    methods: {
        initializeFormValues(){
            this.modified_docs.length = 0
            this.config.disabled = true
        },
        async prepareWorkspace(e) {
            /*Prepare document records to be streamed into a Workspace

            Note: this is separated from the below functionality because `showSaveFilePicker`
            has a limited time until it expires.  This processing may take more than that 
            window of time.
            */

            async function setDocumentsDataArray(doc) {
                const result = await doc.getDataArray()
                const dataArray = result['dataArray']
                const obj = {}
                let idx = 0;
                if(Array.isArray(dataArray)){
                doc.dataArray = dataArray.reduce(function (acc, curr) {
                    acc[idx] = curr
                    idx = idx + 1
                    return acc;
                }, {})
                return doc
                }else if (typeof(dataArray) === 'object' && typeof(dataArray) !== null){
                    doc.dataArray = dataArray
                }
            }

            //this process can be time-consuming
            const promises = toRaw(this.userContentStore.documentsIndex)['documents'].map(async doc => {
                let rec = null
                if (doc.dataArray == null) {
                    rec = await setDocumentsDataArray(doc)
                } else {
                    rec = doc
                }
                return rec
            })
            const modified_docs = await Promise.all(promises)
            this.modified_docs.push(...modified_docs)
            this.config.disabled = false
        },

        async saveWorkStream(e) {

            //TODO:initiate this method for any data change 
            const create = e.target
            const object = {
                documentsIndex: toRaw(this.userContentStore.documentsIndex),
                managedNotes: toRaw(this.userContentStore.managedNotes)
            }
            try {
                const readStream = new Blob([JSON.stringify(object)], { type: 'application/json' }).stream()
                //const compressedStream = readStream.pipeThrough(new TextEncoderStream())    //(new CompressionStream('gzip'))   TODO: I don't know why decoding pipeline fails
                const compressedStream = readStream.pipeThrough(new CompressionStream('gzip'))
                if (!this.config.fileHandle) {
                    this.config.fileHandle = await showSaveFilePicker({
                        suggestedName: ExportAppStateFileName,
                        types: [
                            {
                                description: "GZIP File",
                                accept: {
                                    "application/gzip": [".gz"]
                                }
                            }
                        ]
                    })
                    this.config.fileName = this.config.fileHandle.name
                }
                const writableStream = await this.config.fileHandle.createWritable()
                const result = await compressedStream.pipeTo(writableStream)
                this.$bvModal.hide("save-continue-modal")
            } catch (err) {
                this.resultDisplay.error = `The Workspace Save failed for the following error:&nbsp
                                            ${err.name}: ${err.message}`;
                console.error(err.name, err.message);
            } finally {
                //remove dataArray at the end of the process so that the added data does not reduce performance
                await toRaw(this.userContentStore.documentsIndex)['documents'].forEach((value, key, map) => { value.prepareForIndexDb() })
            }
            //this.$bvModal.hide("save-continue-modal"), note:must be above o/w user may close browser before save is complete
        },


    }
}
</script>


<style scoped>
.fixed-large {
    width: 150px !important;
}

#btnSaveContinue {
    /*margin: 5px;*/
}

.no-li-dot {
    list-style-type: none;
    padding-left: 10px;
    margin-bottom: 0px !important;
}

.no-li-dot label {
    margin: 0px;
}

em {
    font-size: .85rem;
}

input[type="file"] {
    display: none;
}

.custom-file-upload {
    display: inline-block;
    padding: 6px 12px;
    cursor: pointer;
}
</style>