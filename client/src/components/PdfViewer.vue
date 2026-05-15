<template>
    <div style="background-color: black; color: white; text-align: center;">
        <b>{{ record ? record.title : '<no document displayed>' }}</b>
    </div>
    <div id="container" style="background-color: black;">
        <div class="page-navigation">
            <b-button-group size="sm">
                <b-button :disabled="currentPage <= 1" @click="updatePage('start')">&#x21E4</b-button>
                <b-button :disabled="currentPage <= 1" @click="updatePage('decr')">&larr;</b-button>
                <span class="page-btn-grp">{{ currentPage }} / {{ totalPages }}</span>
                <b-button :disabled="currentPage >= totalPages" @click="updatePage('incr')">&rarr;</b-button>
                <b-button :disabled="currentPage >= totalPages" @click="updatePage('end')">&#x21E5</b-button>
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
        margin: '0 auto',
        maxHeight: `calc(100vh - 100px)`
    }">
        <div class="pdf__canvas-layer">
            <canvas ref="canvasLayer" />
        </div>
        <div ref="textLayer" class="pdf__text-layer textLayer"></div>
        <div ref="annotationLayer" class="pdf__annotation-layer annotationLayer"></div>
    </div>
</template>

<script>
import { toRaw } from 'vue'
import { mapStores } from 'pinia'
import { useAppDisplay } from '@/stores/AppDisplay'
import { useUserContent } from '@/stores/UserContent'
//import 'pdfjs-dist/web/pdf_viewer.mjs'
import 'pdfjs-dist/web/pdf_viewer.css'


// Add this helper object globally ABOVE your 'export default' block
const createRawLinkService = (componentInstance) => {
    return {
        baseUrl: null,
        externalLinkTarget: 0,
        externalLinkRel: null,
        externalLinkEnabled: true,
        getDestinationHash: (dest) => "#",
        getAnchorUrl: (hash) => "#",
        addLinkAttributes: (link, url, newWindow = false) => {
            if (link && url) {
                link.href = url;
                if (newWindow) link.target = "_blank";
            }
        },
        setDocument: (doc) => { },
        goToDestination: async (dest) => {
            try {
                const explicitDest = Array.isArray(dest)
                    ? dest
                    : await componentInstance.pdfDocProxy.getDestination(dest);
                if (explicitDest) {
                    const pageIndex = await componentInstance.pdfDocProxy.getPageIndex(explicitDest);
                    await componentInstance.updatePage(pageIndex + 1);
                }
            } catch (e) {
                console.error("Link navigation failed:", e);
            }
        }
    };
};




export default {
    name: 'PdfViewer',
    data() {
        return {
            docId: 1,
            currentPage: 1,
            totalPages: null,
            record: null,
            pdfPageProxy: null,
            pdfDocProxy: null,
            //pageNumPending: null,      //cache waiting page number
            pageRendering: false,   //check conflict

            width: null,
            height: null,
            scale: 1,
            viewportWidth: window.innerWidth,
            viewportHeight: window.innerHeight,
            navHeight: 80,

            userContent: useUserContent(),
            selectedSnippetProcessing: false,
            annotationClickHandler: null,
            selectedSnippetText: null,
            //extractText: true,
        }
    },
    async mounted() {
        //this.renderDisplay()
        window.addEventListener('resize', this.handleResize)
        await this.updateRecord(this.docId);
        await this.processLoadingTask();
    },
    beforeUnmount() {
        window.removeEventListener('resize', this.handleResize)
        this.clearHighlights()
        if (this.pdfDocProxy) {
            toRaw(this.pdfDocProxy).destroy();
        }
        if (this.pdfPageProxy) {
            toRaw(this.pdfPageProxy).destroy();
        }
        if (this.annotationClickHandler && this.$refs.annotationLayer) {
            this.$refs.annotationLayer.removeEventListener("click", this.annotationClickHandler);
        }
    },
    watch: {
        /*
        async currentPage(newValue) {
            await this.updatePage(newValue)
        },*/
        'userContent.selectedSnippet': {
            async handler(newSelectedSnippet, oldValue) {
                if (this.selectedSnippetProcessing) return;
                this.selectedSnippetProcessing = true;
                try {
                    console.log('hi from selectedSnippet!')
                    const snippet = JSON.parse(JSON.stringify(newSelectedSnippet))
                    let page = 1
                    let docId = -1
                    if (snippet.snippet === '') {
                        docId = parseInt(snippet.id)
                        this.selectedSnippetText = null
                        //const docId = parseInt(this.getCurrentRecord.id)
                        //const page = parseInt( snippet.tgtPage )
                        //if(docId != this.docId){
                        await this.updateRecord(docId)
                        await this.processLoadingTask()
                        //this.currentPage = page
                    } else {
                        //if(page != this.currentPage){
                        docId = parseInt(this.userContentStore.selectedDocument)
                        this.selectedSnippetText = snippet.tgtText
                        await this.updateRecord(docId)
                        //await this.processLoadingTask()
                        page = parseInt(snippet.tgtPage)
                        await this.updatePage(page)
                        //this.currentPage = page
                    }
                    //this.currentPage = page
                    //const check = await this.displayHighlightedResultSnippet(newSelectedSnippet)
                    //console.log(`check displayHighlightedResultsItem: ${check}`)
                } finally {
                    this.selectedSnippetProcessing = false;
                }
            },
            deep: true
        },/*
        'userContent.results': {
            async handler(newResults, oldValue) {
                console.log('hi from results!')
                await this.displayAllHighlightedResults(newResults)
            },
            deep: true
        },*/
        currentDocumentSearchResults(newResults, oldValue) {
            if (newResults) {
                this.displayAllHighlightedResults(newResults)
            }
        }
    },
    computed: {
        ...mapStores(useUserContent),
        //changeInStateSelectedSnippet() { return useUserContent.getSelectedSnippet }
        //getCurrentRecord(){ return this.record }
        currentDocumentSearchResults() {
            const docId = this.docId.toString()
            const resultGroups = this.userContentStore.searchTableResults?.resultGroups || []
            return resultGroups.find(group => group.ref === docId) || null
        }
    },
    methods: {
        async handleResize() {
            this.viewportWidth = window.innerWidth
            this.viewportHeight = window.innerHeight
            if (this.pdfDocProxy && this.currentPage) {
                await this.updatePage(this.currentPage);
            }
        },
        calculateViewportScale(pageProxy) {
            const maxHeight = this.viewportHeight - this.navHeight
            return Math.min(this.viewportWidth / pageProxy.view[2], maxHeight / pageProxy.view[3])
            /*
            if(!this.pdfDocProxy) return 1
            const page = this.pdfDocProxy.getPage(this.currentPage)
            const navHeight = 80
            const maxHeight = this.viewportHeight - navHeight
            const maxWidth = this.viewportWidth

            page.then(pageProxy => {
                const viewport = pageProxy.getViewport({ scale: 1})
                const scaleWidth = maxWidth / viewport.width
                const scaleHeight = maxHeight / viewport.height
                this.scale = Math.min(scaleWidth, scaleHeight, 2)
            })
            */
        },
        // utils
        populateDocument(docId) {
            if (!Number.isInteger(docId)) {
                if (!Number.isInteger(this.docId)) {
                    docId = 1
                } else {
                    docId = this.docId
                }
            }
            return this.userContentStore.documentsIndex.documents.filter(item => parseInt(item.id) == docId)[0]
        },
        // page
        async updateRecord(docId) {
            //const records = this.userContentStore.processedFiles
            if (!Number.isInteger(docId)) {
                docId = this.docId
            } else {
                this.docId = docId
            }
            const doc = this.populateDocument(docId)
            const rec = doc
            this.record = rec
            this.currentPage = 1
        },
        async processLoadingTask() {
            if (this.pageRendering) {
                console.warn('Page rendering already in progress');
                return null;
            }
            this.pageRendering = true;
            try {
                return await this._performLoadingTask();
            } finally {
                this.pageRendering = false;
            }
        },
        async _performLoadingTask() {
            try {
                //this.updateRecord()
                const record = this.record
                if (!record) {
                    console.error('no record available to load')
                    return null
                }
                var dataObj = await record.getDataArray()
                //var pdfData = dataObj.record.dataArray
                var pdfData = new Uint8Array(Object.values(dataObj))  //.dataArray ))

                if (this.pdfPageProxy) {
                    this.pdfPageProxy.destroy()
                }
                const loadingTask = await pdfjsLib.getDocument({ data: pdfData, });
                const pdf = await loadingTask.promise;
                this.pdfDocProxy = toRaw(pdf)
                this.totalPages = this.pdfDocProxy.numPages;

                const pageProxy = await toRaw(this.pdfDocProxy).getPage(this.currentPage)
                //this.$refs.pdfLayersWrapper.style.setProperty("--total-scale-factor", `${1}`)
                //const viewport = pageProxy.getViewport({ scale: 1 });
                const { canvasLayer, textLayer, annotationLayer } = this.$refs;

                if (!canvasLayer || !textLayer || !annotationLayer) {
                    console.error('PDF layer refs not properly bound')
                    return null
                }
                //const navHeight = 80
                //const maxHeight = this.viewportHeight - navHeight
                //const scale = Math.min(this.viewportWidth / pageProxy.view[2], maxHeight / pageProxy.view[3])
                const scale = this.calculateViewportScale(pageProxy)
                const viewport = pageProxy.getViewport({ scale })

                this.$refs.pdfLayersWrapper.style.setProperty("--total-scale-factor", `${scale}`)
                //this.renderText(pageProxy, textLayer, viewport);
                await this.renderAnnotations(pageProxy, annotationLayer, viewport);
                await this.renderText(pageProxy, textLayer, viewport);
                const results = await this.renderCanvas(pageProxy, canvasLayer, viewport);
                if (this.currentDocumentSearchResults) {
                    await this.displayAllHighlightedResults(this.currentDocumentSearchResults)
                }
                return results;
            } catch (error) {
                console.error('Error loading PDF:', error)
                return null;
            }
        },
        async updatePage(page_or_direction) {
            if (this.pageRendering) {
                console.warn('Page rendering already in progress');
                return false;
            }
            this.pageRendering = true;
            try {
                return await this._performPageUpdate(page_or_direction);
            } finally {
                this.pageRendering = false;
            }
        },
        async _performPageUpdate(page_or_direction) {
            try {
                let page = null
                if (Number.isInteger(page_or_direction)) {
                    page = page_or_direction

                } else if (page_or_direction == 'end') {
                    page = this.totalPages

                } else if (page_or_direction == 'incr') {
                    if (this.currentPage == this.totalPages) {
                        console.warn('End of pages reached')
                        return false
                    } else {
                        page = this.currentPage + 1
                    }
                } else if (page_or_direction == 'decr') {
                    if (this.currentPage == 1) {
                        console.warn('Beginning of pages reached')
                        return false
                    } else {
                        page = this.currentPage - 1
                    }
                } else if (page_or_direction == 'start') {
                    page = 1
                }
                if (this.pdfPageProxy) {
                    this.pdfPageProxy.destroy()
                }
                if (!this.record) {
                    console.error('No record available')
                    return false
                }
                const dataObj = await this.record.getDataArray();
                const pdfData = new Uint8Array(Object.values(dataObj))
                /*
                //this.currentPage = page
                var dataObj = await this.record.getDataArray()
                //var pdfData = dataObj.record.dataArray
                //var pdfData = dataObj.dataArray
                //var pdfData = new Uint8Array(Object.values( dataObj.dataArray ))
                var pdfData = new Uint8Array(Object.values( dataObj ))
                */
                const loadingTask = await pdfjsLib.getDocument({ data: pdfData, });
                const pdf = await loadingTask.promise;
                this.pdfDocProxy = toRaw(pdf)
                const pageProxy = await toRaw(this.pdfDocProxy).getPage(page);
                this.currentPage = page
                this.totalPages = this.pdfDocProxy.numPages;
                const { canvasLayer, textLayer, annotationLayer } = this.$refs;
                if (!canvasLayer || !textLayer || !annotationLayer) {
                    console.error('PDF layer refs not properly bound')
                    return false
                }
                //const navHeight = 80
                //const maxHeight = this.viewportHeight - navHeight
                //const scale = Math.min(this.viewportWidth / pageProxy.view[2], maxHeight / pageProxy.view[3])
                const scale = this.calculateViewportScale(pageProxy)
                const viewport = pageProxy.getViewport({ scale });
                //this.renderText(pageProxy, textLayer, viewport);
                await this.renderAnnotations(pageProxy, annotationLayer, viewport);
                await this.renderText(pageProxy, textLayer, viewport);
                await this.renderCanvas(pageProxy, canvasLayer, viewport);


const textLayerContainer = this.$refs.textLayer;
if (textLayerContainer) {
  textLayerContainer.innerHTML = "";
  
  // Explicitly assign width and height wrappers matching pixel specs
  textLayerContainer.style.width = `${viewport.width}px`;
  textLayerContainer.style.height = `${viewport.height}px`;

  // CRITICAL FIX: Pass the exact numeric scaling coefficient to the CSS layout engine
  textLayerContainer.style.setProperty('--scale-factor', viewport.scale);

  const textContent = await pageProxy.getTextContent();

  const textLayer = new pdfjsLib.TextLayer({
    textContentSource: textContent,
    container: textLayerContainer,
    viewport: viewport,
  });

  await textLayer.render();
}








                //TODO: await this.displayAllHighlightedResults()
                if (this.currentDocumentSearchResults) {
                    await this.displayAllHighlightedResults(this.currentDocumentSearchResults)
                }
                return true
            } catch (error) {
                console.error('Error updating page:', error)
            }
        },

        // layers
        async renderText(pdfPageProxy, textLayerContainer, viewport) {
            try {
                textLayerContainer.replaceChildren()
                const content = await pdfPageProxy.getTextContent()
                const renderTask = new pdfjsLib.TextLayer({
                    container: textLayerContainer,
                    textContentSource: content,
                    viewport: viewport.clone({ dontFlip: true })
                });
                await renderTask.render();
            } catch (error) {
                console.error('Error rendering text layer:', error)
            }
        },
        async renderCanvas(pdfPageProxy, canvasLayer, viewport) {
            try {
                const { width, height, rotation } = viewport;
                this.width = width;
                this.height = height;
                canvasLayer.width = width;
                canvasLayer.height = height;
                const context = canvasLayer.getContext("2d");
                if (!context) {
                    console.error('Failed to get 2D context from canvas')
                    return null
                }
                const renderContext = {
                    canvasContext: context,
                    viewport: viewport
                };
                return await pdfPageProxy.render(renderContext);
            } catch (error) {
                console.error('Error rendering canvas:', error)
            }
        },
        async getAnnotations(pageProxy) {
            const annotations = await pageProxy.getAnnotations({ intent: "display" });
            return annotations;
        },
        async displayAllHighlightedResults(searchResults) {
            if (!searchResults || !searchResults.phrase || searchResults.phrase.length === 0) {
                this.clearHighlights()
                return
            }
            if (!this.record || !this.$refs.textLayer) {
                return
            }
            const currentPageText = this.record.body_pages[this.currentPage]
            if (!currentPageText) {
                return
            }
            this.clearHighlights()
            const textLayer = this.$refs.textLayer
            const spans = textLayer.querySelectorAll('span')
            searchResults.phrase.forEach(phraseToFind => {
                const searchRegex = new RegExp(phraseToFind.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi')
                spans.forEach(span => {
                    const text = span.textContent
                    if (searchRegex.test(text)) {
                        const highlightedHTML = text.replace(searchRegex, match =>
                            `<mark class="pdf-hgighligh">${match}</mark>`
                        )
                        span.innerHTML = highlightedHTML
                    }
                })
            })
            if (this.selectedSnippetText) {
                this.markSelectedSnippet()
            }
        },
        clearHighlights() {
            if (!this.$refs.textLayer) return
            const highlights = this.$refs.textLayer.querySelectorAll('mark.pdf-highlight')
            highlights.forEach(mark => {
                const parent = mark.parentNode
                parent.textContent = parent.textContent
            })

        },
        markSelectedSnippet() {
            if (!this.$refs.textLayer || !this.selectedSnippetText) return
            const highlights = this.$refs.textLayer.querySelectorAll('mark.pdf-highlight')
            const normalizedSelectedText = this.selectedSnippetText.trim().toLowerCase()
            highlights.forEach(mark => {
                const markText = mark.textcontent.trim().toLowerCase()
                if (markText === normalizedSelectedText) {
                    mark.classList.add('selected')
                }
            })
        },
        createAnnotationClickHandler(annotations) {
            return async (event) => {
                try {
                    let annotationTarget = event.target.parentNode;
                    if (!annotationTarget) {
                        return;
                    }
                    const id = annotationTarget.dataset.annotationId;
                    if (!id) {
                        return;
                    }
                    const annotationLinkId = annotations.find((ele) => ele.id == id);
                    if (!annotationLinkId) {
                        return;
                    }
                    if (!this.pdfDocProxy) {
                        console.error('PDF document not loaded')
                    }
                    const pageIndex = await this.pdfDocProxy.getPageIndex(
                        annotationLinkId.dest[0]
                    );
                    await this.updatePage(pageIndex + 1);
                } catch (error) {
                    console.error('Error handling annotation click:', error)
                }
            };
        },
        async renderAnnotations(pdfPageProxy, annotationLayerContainer, viewport) {
            try {
                // The below can only be solved by updating to the newest version of pdfjslib
                // Clear container
                // 1. Clear any existing child DOM nodes cleanly
                annotationLayerContainer.innerHTML = "";

                const rawDoc = toRaw(this.pdfDocProxy);
                const rawPage = toRaw(pdfPageProxy);
                const annotations = await rawPage.getAnnotations();

                // Create a functional handler wrapper to retain target extraction data
                const linkService = {
                    // 1. Maintain mandatory configuration properties for the parser engine
                    baseUrl: null,
                    externalLinkTarget: 2, // 2 maps internally to '_blank' targeting rules
                    externalLinkRel: 'noopener noreferrer',
                    externalLinkEnabled: true,

                    // 2. Map actual link attributes rather than dropping them
                    addLinkAttributes: (link, url, newWindow = true) => {
                        if (link && url) {
                            link.href = url;
                            link.rel = 'noopener noreferrer';
                            if (newWindow || linkService.externalLinkTarget === 2) {
                                link.target = "_blank";
                            }
                        }
                    },

                    // 3. Keep track of tracking hashes dynamically 
                    getDestinationHash: (dest) => {
                        if (typeof dest === 'string') return `#${dest}`;
                        return `#${JSON.stringify(dest)}`;
                    },

                    getAnchorUrl: (hash) => hash || "#",

                    // 4. Resolve internal jump targets dynamically via the document context
                    goToDestination: async (dest) => {
                        try {
                            let explicitDest = dest;
                            if (typeof dest === 'string') {
                                explicitDest = await rawDoc.getDestination(dest);
                            }

                            if (explicitDest) {
                                // Look up the explicit page index mapping via the document pointer
                                const pageIndex = await rawDoc.getPageIndex(explicitDest[0]);
                                // Jump your Vue component to the calculated destination page
                                if (typeof this.updatePage === 'function') {
                                    await this.updatePage(pageIndex + 1);
                                }
                            }
                        } catch (e) {
                            console.error("Internal link navigation failed:", e);
                        }
                    }
                };

                // Build the annotation layer using the functional adapter wrapper
                const annotationLayer = new pdfjsLib.AnnotationLayer({
                    viewport: viewport.clone({ dontFlip: true }),
                    div: annotationLayerContainer,
                    annotations: annotations,
                    page: rawPage,
                    linkService: linkService,
                    imageResourcesPath: '/images/'
                });


                await annotationLayer.render({
                    viewport: viewport.clone({ dontFlip: true }),
                    div: annotationLayerContainer,
                    annotations: annotations,
                    page: rawPage,
                    linkService: linkService,
                });
            } catch (error) {
                console.error('Error rendering annotation layer:', error)
            }
        },
    }
}
</script>



<style scoped>
.page-btn-grp {
    color: white;
    padding-left: 20px;
    padding-right: 20px;
}



#container {
    font-family: Avenir, Helvetica, Arial, sans-serif;
    text-align: center;
    color: #2c3e50;
    /*margin-top: 60px;*/
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
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: auto;

    .pdf__page-wrapper {
      position: relative; /* Isolates absolutely positioned child layers */
      display: inline-block; /* CRITICAL: Prevents margin collapse and horizontal layout shifting */
      vertical-align: top;
      overflow: hidden;
    }

    .pdf__canvas-layer {
        display: block;
        max-width: 100%;
        position: absolute;
        inset: 0;
        z-index: 1;
    }

    .pdf__text-layer {
        inset: 0;
        position: absolute;
        opacity: 1;
        line-height: 1;
        transform-origin: 0% 0%; /* Guarantees scaling math anchors to the top-left edge corner */
        z-index: 2;
        pointer-events: auto; /* Required to allow highlight mouse dragging */

        /* Ensure child text segments do not block rendering flow */
        :deep(.textLayer) {
          position: absolute;
          text-align: initial;
          inset: 0;
          overflow: hidden;
          line-height: 1;
          pointer-events: auto;        /* CRITICAL: Instructs the browser to listen for text marking */
          user-select: text !important; /* Forces system text cursor behavior */
          -webkit-user-select: text !important;
        }

        /* CRITICAL HIGHLIGHT FIX: Explicitly color the background selection canvas */
        :deep(.textLayer *::selection) {
          background-color: rgba(0, 0, 255, 0.25) !important; /* Semi-transparent blue highlight */
          color: transparent !important;                      /* Keeps text invisible to prevent rendering artifact blur */
        }       

        /* Support fallback rendering engines across older webkit configurations */
        :deep(.textLayer *::-moz-selection) {
          background-color: rgba(0, 0, 255, 0.25) !important;
          color: transparent !important;
        }

        :deep(.textLayer span) {
          color: transparent;
          position: absolute;
          white-space: pre;
          cursor: text !important;     /* Forces the text I-beam selection cursor */
          transform-origin: 0% 0%; /* Prevents individual font glyph shifting when selected */
          pointer-events: auto;        /* Enables targeting of individual letter glyphs */
        }

        br::selection {
            color: transparent;
        }

        :deep(span) {
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

    :deep(.pdf-highlight) {
        background-color: rgba(255, 255, 0, 0.5);
        color: black;
        font-weight: bold;
        padding: 0;
        margin: 0;
    }

    :deep(.pdf-highlight.selected) {
        background-color: orange;
    }

    .pdf__annotation-layer {
        inset: 0;
        position: absolute;
        pointer-events: none;
        z-index: 3 !important;
        pointer-events: none;        /* CRITICAL: Clicks pass directly through empty zones to the text layer */

        /* Re-enable click actions ONLY for the actual clickable hyperlink tags */
        :deep(.annotationLayer .linkAnnotation a) {
          pointer-events: auto !important;
          cursor: pointer;
        }

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