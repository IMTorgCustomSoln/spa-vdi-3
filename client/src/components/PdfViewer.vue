<template>
    <div style="background-color: black; color: white; text-align: center;">
        <b>{{ this.getCurrentDoc ? this.getCurrentDoc.title : '<no document displayed>' }}</b>
    </div>
    <div style="background-color: black;">
        <b-button-group size="sm">
            <b-button @click="loadDoc">Load Selected Doc</b-button>
            <b-button @click="highlightText">Hightlight Text</b-button>
            <b-button @click="extractTextRadio">Select Text ({{ formatBoolean(this.extractText) }})</b-button>
            <b-button @click="extractImageRadio" :disabled="true">Select Image ({{ formatBoolean(this.extractImage)
                }})</b-button>
        </b-button-group>
    </div>

    <div id="pageContainer" @srcData="">
        <b-embed type="iframe" aspect="4by5" allowfullscreen ref="frame" id="pdf-js-viewer" title="IFrameViewer"
            frameborder="0" @load="iframeLoaded" :src="getPath" class="object-fit-none border rounded">
        </b-embed>
        <!--
        <iframe ref="frame" id="pdf-js-viewer" title="IFrameViewer" frameborder="0" @load="iframeLoaded" :src="getPath"
            class="object-fit-none border rounded"
            width="100%" height="100%">
        </iframe> --> <!--these do not appear necessary: sandbox="allow-same-origin allow-scripts"   -->
    </div>
</template>


<script>
import { toRaw } from 'vue'
import { mapStores } from 'pinia'
import { useAppDisplay } from '@/stores/AppDisplay'
import { useUserContent } from '@/stores/UserContent'


export default {
    name: "PdfViewer",
    watch: {
        //TODO, note: event source is Table Snippets
        'userContentStore.selectedSnippet': {
            handler: async function (newVal, oldVal) {
                let pg = 1
                let tgtText = ''
                if (newVal.snippet != '') {
                    /*
                    //const txtPg = parseInt(newVal.snippet.split('<b>pg.')[1].split('|')[0])
                    const txtPg = newVal.tgtPage
                    pg = txtPg <= 1 ? txtPg : txtPg - 1
                    //tgtText = newVal.snippet.split('<b style="background-color: yellow">')[1].split('</b>')[0]
                    tgtText = newVal.tgtText*/
                }
                const app = await this.getApp
                const currentDoc = this.getCurrentDoc
                if( parseInt(currentDoc.id) != JSON.parse(JSON.stringify(newVal)).id ){
                    await this.loadDoc()
                    //await this.chgToPg3()
                }
                this.search(newVal.tgtText)
                try {
                    console.log(newVal.tgtPage)
                    if (Number.isInteger(newVal.tgtPage)){
                        app.page = parseInt(newVal.tgtPage)     //TODO:TypeError: Cannot destructure property 'div' of 'pageView' as it is undefined.  ans) https://github.com/VadimDez/ng2-pdf-viewer/issues/224#issuecomment-485322711
                    } /*else {
                        app.page = 1
                    }*/
                } catch (error) {
                    console.log('TODO: this is a known bug in mozilla pdfjs-dist.  It will be solved with a future release')
                    console.log(error)
                }
            }
        },
    },
    data() {
        return {
            docBlob: null,
            docPath: null,
            pathViewer: '/pdfjs-dist/web/viewer.html',     //'https://cdn.jsdelivr.net/gh/IMTorgOpenDataTools/pdfjs-dist@master/web/viewer.html'
            query: '?file=',
            pathFile: null,     //'./compressed.tracemonkey-pldi-09.pdf',      //'../../../tests/data/10469527483063392000-cs_nlp_2301.09640.pdf',    //must be relative to `viewer.html` location

            newNote: null,
            currentDocumentId: 0,
            extractText: false,
            extractImage: false
        }
    },
    computed: {
        ...mapStores(useAppDisplay, useUserContent),
        getPath() {
            return this.pathViewer + this.query //+ this.pathFile //<<< default pdf
        },
        async getApp() {
            const app = await document.getElementById('pdf-js-viewer').contentWindow.PDFViewerApplication
            return app
        },
        getDocument() {
            const docId = this.userContentStore.getSelectedDocument
            return this.userContentStore.documentsIndex.documents.filter(item => item.id == docId)[0]         //TODO:must use the Table array that is sorted on Score o/w incorrect
        },
        getCurrentDoc() {
            const id = this.currentDocumentId ? this.currentDocumentId : 0
            return this.userContentStore.documentsIndex.documents.filter(item => item.id == id)[0]
        }
    },
    methods: {
        formatBoolean(text) { return text == true ? 'on' : 'off' },
        // Listener Logic
        //listener and selector for user-selected text
        async iframeLoaded() {
            await this.getApp
            await this.loadDoc()
            return true
        },
        async extractTextRadio() {
            const app = await this.getApp
            const doc = document.getElementById('pdf-js-viewer').contentWindow.document
            this.extractText = !this.extractText
            if (this.extractText) {
                doc.addEventListener('selectionchange', this.logTextToNotesManager)
            } else {
                doc.removeEventListener('selectionchange', this.logTextToNotesManager)
                const reference = `<div style="font-weight: bold">${this.getDocument.filepath}, pg.${app.page} |</div> ${this.newNote}`
                const noteItem = {
                    id: Date.now(),
                    list: 'stagingNotes',
                    type: 'auto',
                    page: app.page,
                    file: this.getDocument.filepath,
                    text: this.newNote,
                    begin: this.getDocument.body_pages[1].indexOf(this.newNote),
                    calibrateSubString: this.getDocument.body_pages[app.page].substring(0,100),
                    calibratePageLength: this.getDocument.body_chars[app.page],
                    innerHTML: reference,
                    innerText: reference
                }
                this.userContentStore.newNote = noteItem
                this.userContentStore.addNewNoteToManager()
                this.highlightText()
            }
            return true
        },
        logTextToNotesManager(e) {
            const txt = this.getSelectedText(e)
            this.newNote = txt
        },
        getSelectedText(e) {
            const iframeWindow = document.getElementById('pdf-js-viewer').contentWindow.document
            if (iframeWindow) {
                return iframeWindow.getSelection().toString()
            }
            else if (document.selection) {
                return document.selection.createRange().text
            }
            return '';
        },
        extractImageRadio() {
            // TODO: this fails !!!
            //google: pdfjs pdfviewer area selector
            //ref: https://stackoverflow.com/questions/17703906/use-a-canvas-to-select-a-portion-of-a-pdf-document?rq=3
            //ref: https://stackoverflow.com/questions/49605369/highlight-area-in-pdfjs-canvas-based-on-native-pdf-coordinates
            //ref: https://stackoverflow.com/questions/55411493/pdf-js-with-canvas-draw-draw-rectangle-in-canvas-with-pdf-loaded-using-pdf-js?noredirect=1&lq=1
            //ref: https://stackoverflow.com/questions/58590845/draw-rectangle-in-canvas-with-loaded-pdf-file-using-pdf-js?noredirect=1&lq=1
            this.extractImage = !this.extractImage

            const iframeWindow = document.getElementById('pdf-js-viewer').contentWindow
            const app = document.getElementById('pdf-js-viewer').contentWindow.PDFViewerApplication
            const pageIndex = app.page - 1

            const _page = app.pdfViewer._pages[pageIndex]
            const canvas = _page.canvas.getContext("2d")
            //var x1 = 0, y1 = 0, x2 = 0, y2 = 0
            if (this.extractImage) {
                iframeWindow.addEventListener('onmousedown', this.onmousechange)
                iframeWindow.addEventListener('onmouseup', this.onmousechange)
            } else {
                iframeWindow.removeEventListener('onmousedown', this.onmousechange)
                iframeWindow.removeEventListener('onmouseup', this.onmousechange)
            }
        },


        onmousechange(e) {
            //div.hidden = 0;
            const tmp = window.getSelection().toString()
            console.log(tmp)
            let x1 = 0
            let y1 = 0
            let x2 = e.clientX;
            let y2 = e.clientY;
            console.log(`x1:${x1},y1: ${y1}    x2:${x2},y2: ${y2}`)
            //canvas.removeEventListener('onmousedown', (e) => {onmousedown(e)})
            //reCalc();
            /*
        }
        onmousemove = function(e) {
            x2 = e.clientX;
            y2 = e.clientY;
            console.log(`x1:${x1},y1: ${y1}    x2:${x2},y2: ${y2}`)
            canvas .removeEventListener('onmousemove', (e) => {onmousemove(e)})
            //reCalc();
        }*
        function onmouseup(e) {
            //div.hidden = 1;
            console.log(`x1:${x1},y1: ${y1}    x2:${x2},y2: ${y2}`)
            canvas.removeEventListener('onmouseup', (e) => {onmouseup(e)})
        }*/

        },



        // Button Logic
        search(tgtText) {
            //search and highlight text
            const app = document.getElementById('pdf-js-viewer').contentWindow.PDFViewerApplication
            const searchObj = {
                query: tgtText,
                highlightAll: true,
                caseSensitive: false,
                findPrevious: undefined,
                phraseSearch: true
            }
            app.pdfViewer.eventBus.dispatch('find', searchObj)

        },
        async saveDoc() {
            //get document properties
            const app = document.getElementById('pdf-js-viewer').contentWindow.PDFViewerApplication
            const doc = await app.pdfDocument
            console.log(`Get numPages: ${doc.numPages}`)
            const blob = await doc.getData()
            this.docBlob = blob
            console.log(`Get Uint8Array (first 50...): ${blob.slice(0, 50)}`)
        },
        async loadDoc() {
            //load document from typed array
            //expected workflow: click save doc, open a new document (manually) using `Open File` button, then click load doc 
            const app = await this.getApp  //document.getElementById('pdf-js-viewer').contentWindow.PDFViewerApplication
            const doc = this.getDocument
            if (doc.id != this.currentDocumentId) {
                try {
                    const dataArray = await toRaw(doc.getDataArray())
                    const tgt = { data: Object.values(dataArray.dataArray) }
                    await app.open(tgt)
                    app.page = 1
                } catch (error) {
                    console.log(error)
                    /*
                    alert(
                        `*** ALERT: Please press 'OK' then reload the page. ***\n\nThis error is caused by an unavailable 'worker' provided by the server.  It will be solved in a future release.
                        `
                    )*/
                }
                this.currentDocumentId = doc.id
            }
        },
        async chgToPg3() {
            const app = document.getElementById('pdf-js-viewer').contentWindow.PDFViewerApplication
            app.page = 3
            await app.page
            //app.zoomIn(5)
        },



        // Logic
        highlightText() {
            const selected = this.getSelectionCoords()
            this.showHighlight(selected)
        },
        getSelectionCoords() {
            const iframeWindow = document.getElementById('pdf-js-viewer').contentWindow
            const app = document.getElementById('pdf-js-viewer').contentWindow.PDFViewerApplication
            const pageIndex = app.page - 1

            const _page = app.pdfViewer._pages[pageIndex]
            const pageRect = _page.canvas.getClientRects()[0]
            if (iframeWindow.getSelection().rangeCount == 0) {
                console.log('ERROR: no text selected')
                return {}
            }
            if (iframeWindow.getSelection().rangeCount > 1) {
                console.log('ERROR: only a single, continuous text may be selected')
                return {}
            }
            const selectionRects = iframeWindow.getSelection().getRangeAt(0).getClientRects()
            const viewport = _page.viewport
            const selectionRectsList = Object.values(selectionRects)
            const selected = selectionRectsList.map(function (r) {
                return viewport.convertToPdfPoint(r.left - pageRect.x, r.top - pageRect.y).concat(
                    viewport.convertToPdfPoint(r.right - pageRect.x, r.bottom - pageRect.y));
            })    // left, top, right, bottom
            return { pageIndex: pageIndex, coordinates: selected }      //only allow a single, continuous selection
        },
        showHighlight({ pageIndex, coordinates }) {
            const app = document.getElementById('pdf-js-viewer').contentWindow.PDFViewerApplication
            const _page = app.pdfViewer._pages[pageIndex]
            const viewport = _page.viewport

            coordinates.forEach(function (rect) {
                let highlightColor = 'ff990080'   //generateColor();    transparency of 80%
                let bounds = viewport.convertToViewportRectangle(rect);

                var x1 = Math.min(bounds[0], bounds[2]);
                var y1 = Math.min(bounds[1], bounds[3]);
                var width = Math.abs(bounds[0] - bounds[2]);
                var hight = Math.abs(bounds[1] - bounds[3]);

                var el = createRectDiv([x1, y1, width, hight], highlightColor);
                _page.textLayer.div.appendChild(el)
            }, this)
        },

        getExtractedImage() {
            const app = document.getElementById('pdf-js-viewer').contentWindow.document//.PDFViewerApplication

        },
        getSelectedImage() {
            //TODO task: select image (such as graph) - is this possible?
            //ref: https://stackoverflow.com/questions/13416800/how-to-generate-an-image-from-imagedata-in-javascript
            //ref: https://stackoverflow.com/questions/923885/capture-html-canvas-as-gif-jpg-png-pdf?noredirect=1&lq=1
            const iframeWindow = document.getElementById('pdf-js-viewer').contentWindow.document
            const app = document.getElementById('pdf-js-viewer').contentWindow.PDFViewerApplication
            const pageIndex = app.page - 1
            const _page = app.pdfViewer._pages[pageIndex]
            _page.canvas.getContext('2d').getImageData(1, 1, 1, 1)     //x, y, width, height

        },
    }
}




const generateColor = () => {
    return Math.floor(Math.random() * 16777215).toString(16)
}

function createRectDiv(boundBox, highlightColor) {
    // console.log(randomColor);
    var el = document.createElement('div');
    el.setAttribute('class', 'hiDiv')
    el.setAttribute('style', 'position: absolute; background-color: #' + highlightColor + '; opacity: 0.95;' +
        'left:' + boundBox[0] + 'px; top:' + boundBox[1] + 'px;' +
        'width:' + boundBox[2] + 'px; height:' + boundBox[3] + 'px;');
    return el;
}






</script>

<style scoped>
/*
#pageContainer {
    /*position: absolute;
    height: 90vh;
    height: 100%;
}*/
#pdf-js-viewer {
    max-width: 100%;
}

#div {
    border: 1px dotted #000;
    position: absolute;
}


/*
NOTE: this changes the sizing based on aspect-ratio
ref: https://stackoverflow.com/questions/78324211/how-to-create-custom-video-embed-ratios-in-bootstrap
*/
.embed-responsive-4by5 {
    padding-top: 115%;
}

/*
#cloud_main_page{
  width:400px;
  height:400px;
  
  position:relative;
}
#pageContainer #pdf-js-viewer{	
	position:absolute;
	
	background-color:rgba(6, 217, 160, 0.05);
	border: 1px solid rgba(6, 217, 160, 0.3);	
}*/
</style>