<template>
    <div style="background-color: black; color: white; text-align: center;">
        <b>{{ this.getCurrentDoc ? this.getCurrentDoc.title : '<no document displayed>' }}</b>
    </div>
    <div id="container" style="background-color: black;">
        <div class="page-navigation">
            <b-button-group size="sm">
                <b-button :disabled="currentPage <= 1" @click="updatePage('decr')">&larr;</b-button>
                <span class="page-btn-grp">{{ currentPage }}/{{ totalPages }}</span>
                <b-button :disabled="currentPage >= totalPages" @click="updatePage('incr')">&rarr;</b-button>
                <!--
                <b-button @click="extractTextRadio">Select Text ({{ formatBoolean(this.extractText) }})</b-button>
                <b-button @click="extractImageRadio" :disabled="true">Select Image ({{ formatBoolean(this.extractImage) }})</b-button>
                -->
            </b-button-group>
        </div>
    </div>

    <div ref="pdfLayersWrapper" class="pdf__layers" :style="{
        height: `${height}px`,
        width: `${width}px`,
        border: '1px solid #dfdfdf',
        margin: '0 auto'
    }">
        <div class="pdf__canvas-layer">
            <canvas ref="canvasLayer" />
        </div>
        <div ref="textLayer" class="pdf__text-layer"></div>
        <div ref="annotationLayer" class="pdf__annotation-layer"></div>
    </div>
</template>

<script>
import { toRaw } from 'vue'
import { mapStores } from 'pinia'
import { useAppDisplay } from '@/stores/AppDisplay'
import { useUserContent } from '@/stores/UserContent'

export default {
    name: 'PdfViewer',
    data() {
        return {
            record: null,
            //pdfDocProxy: null,
            pdfPageProxy: null,
            //pageSelection: 1,
            currentPage: 1,
            totalPages: null,

            width: null,
            height: null,

            extractText: true,
            userContent: useUserContent()
        }
    },
    async mounted() {
        //this.renderDisplay()
        await this.processLoadingTask();
    },
    watch: {
        async currentPage(newValue) {
            await this.updatePage(newValue)
        },
        'userContent.selectedSnippet': {
            async handler(newSelectedSnippet, oldValue) {
                console.log('hi from selectedSnippet!')
                const docId = parseInt( JSON.parse(JSON.stringify(newSelectedSnippet)).id )
                await this.updateRecord(docId)
                await this.processLoadingTask()
                //const check = await this.displayHighlightedResultSnippet(newSelectedSnippet)
                //console.log(`check displayHighlightedResultsItem: ${check}`)
            },
            deep: true
        },
        'userContent.results': {
            async handler(newResults, oldValue) {
                console.log('hi from results!')
                await this.displayAllHighlightedResults(newResults)
            },
            deep: true
        },
    },
    computed: {
        ...mapStores(useUserContent),
        //changeInStateSelectedSnippet() { return useUserContent.getSelectedSnippet }
    },
    methods: {
        // page
        async updateRecord(docId) {
            //const records = this.userContentStore.processedFiles
            if (docId == undefined){
                docId = 1
            }
            const doc = this.userContentStore.documentsIndex.documents.filter(item => item.id == docId)[0] 
            const rec = doc
            this.record = rec
        },
        async processLoadingTask() {
            this.updateRecord()
            const record = this.record
            if (!record) { return null }
            var dataObj = await record.getDataArray()
            //var pdfData = dataObj.record.dataArray
            var pdfData = dataObj.dataArray

            const loadingTask = await pdfjsLib.getDocument({ data: pdfData, });
            const pdf = await loadingTask.promise;
            this.pdfDocProxy = pdf
            this.totalPages = this.pdfDocProxy.numPages;

            const pageProxy = await toRaw(this.pdfDocProxy).getPage(this.currentPage)
            this.$refs.pdfLayersWrapper.style.setProperty("--total-scale-factor", `${1}`)
            const viewport = pageProxy.getViewport({ scale: 1 });
            const { canvasLayer, textLayer, annotationLayer } = this.$refs;

            this.renderText(pageProxy, textLayer, viewport);
            this.renderAnnotations(pageProxy, annotationLayer, viewport);
            return this.renderCanvas(pageProxy, canvasLayer, viewport);
        },
        async updatePage(page) {
            const pageProxy = await this.pdfDocProxy.getPage(page);
            const { canvasLayer, textLayer, annotationLayer } = this.$refs;
            const viewport = pageProxy.getViewport({ scale: 1 });

            this.renderText(pageProxy, textLayer, viewport);
            this.renderAnnotations(pageProxy, annotationLayer, viewport);
            this.renderCanvas(pageProxy, canvasLayer, viewport);
            await this.displayAllHighlightedResults()
            return true
        },


        // layers
        async renderText(pdfPageProxy, textLayerContainer, viewport) {
            textLayerContainer.replaceChildren()
            const content = await pdfPageProxy.getTextContent()
            const renderTask = new pdfjsLib.TextLayer({
                container: textLayerContainer,
                textContentSource: content,
                viewport: viewport.clone({ dontFlip: true })
            });
            await renderTask.render();

        },
        async renderCanvas(pdfPageProxy, canvasLayer, viewport) {
            const { width, height, rotation } = viewport;
            this.width = width;
            this.height = height;
            canvasLayer.width = width;
            canvasLayer.height = height;
            const context = canvasLayer.getContext("2d");
            const renderContext = {
                canvasContext: context,
                viewport: viewport
            };
            return pdfPageProxy.render(renderContext);
        },
        async getAnnotations(pageProxy) {
            const annotations = await pageProxy.getAnnotations({ intent: "display" });
            return annotations;
        },
        async renderAnnotations(pdfPageProxy, annotationLayerContainer, viewport) {
            annotationLayerContainer.replaceChildren();
            annotationLayerContainer.width = this.width;
            annotationLayerContainer.height = this.height;
            const annotations = await this.getAnnotations(pdfPageProxy);
            const clonedViewport = viewport.clone({ dontFlip: true });
            const annotationLayer = new pdfjsLib.AnnotationLayer({
                div: annotationLayerContainer,
                accessibilityManager: undefined,
                annotationCanvasMap: undefined,
                annotationEditorUIManager: undefined,
                page: pdfPageProxy,
                viewport: clonedViewport,
                /* new pdfjs-dist@4.10.38 */
                structTreeLayer: null
            });
            await annotationLayer.render({
                div: annotationLayerContainer,
                viewport: clonedViewport,
                page: pdfPageProxy,
                annotations,
                imageResourcesPath: undefined,
                renderForms: false,
                linkService: new pdfjsViewer.SimpleLinkService(),
                downloadManager: null,
                annotationStorage: undefined,
                enableScripting: false,
                hasJSActions: undefined,
                fieldObjects: undefined
            });
            annotationLayerContainer.addEventListener("click", async (event) => {
                let annotationTarget = event.target.parentNode;
                if (!annotationTarget) {
                    return;
                }
                const id = annotationTarget.dataset.annotationId;
                if (!id) {
                    return;
                }
                const annotationLinkId = annotations.find((ele) => ele.id === id);
                if (!annotationLinkId) {
                    return;
                }
                const pageIndex = await this.pdfDocProxy.getPageIndex(
                    annotationLinkId.dest[0]
                );
                this.currentPage = pageIndex + 1;
            });
        },
    }
}
</script>



<style scoped>
.page-btn-grp {
    padding-left: 20px;
    padding-right: 20px;
}



#container {
    font-family: Avenir, Helvetica, Arial, sans-serif;
    text-align: center;
    color: #2c3e50;
    margin-top: 60px;
}

a,
button,
.badge {
    color: #4fc08d;
}

button,
.badge {
    background: none;
    border: solid 1px;
    border-radius: 2em;
    font: inherit;
    padding: 0.75em 2em;
}

.badge {
    display: inline-block;
    margin-bottom: 1rem;
    margin-top: 1rem;
}

/* Note: layers will fail without proper css
annotationLayer must be on top | index: 6 */
.pdf__layers {
    position: relative;

    .pdf__canvas-layer {
        position: absolute;
        inset: 0;
    }

    .pdf__text-layer {
        inset: 0;
        position: absolute;
        opacity: 1;
        line-height: 1;
        z-index: 5;

        br::selection {
            color: transparent;
        }

        span {
            color: transparent;
            cursor: text;
            position: absolute;
            transform-origin: 0% 0%;
            white-space: pre;

            &::selection {
                background-color: black;
                color: yellow;
            }
        }
    }

    .pdf__annotation-layer {
        inset: 0;
        position: absolute;
        pointer-events: none;
        z-index: 6 !important;

        section {
            position: absolute;
            text-align: initial;
            pointer-events: auto;
            box-sizing: border-box;

            &:not(.popupAnnotation) {
                z-index: 6 !important;
            }

            &:has(div.annotationContent) {
                canvas.annotationContent {
                    display: none;
                }
            }

            a {
                height: 100%;
                left: 0;
                position: absolute;
                top: 0;
                width: 100%;
                cursor: pointer;

                &:hover {
                    background-color: rgba(99, 39, 245, 0.3);
                }
            }
        }
    }
}
</style>