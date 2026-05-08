<template>
    <!-- Open modal button -->
    <b-button
         id='btnAbout' 
         v-b-modal="'about-modal'"
         variant="primary"
         class=""
         >
        <b-icon icon="gear-fill" aria-hidden="true"></b-icon>
     </b-button>
 
     <!-- modal -->
     <b-modal id='about-modal' ok-only>
        <template #modal-title>
             Settings
        </template>
        <div>
          <b-tabs content-class="mt-3">
            <b-tab title="AI" active>
                <p>
                    * Add progress bar of models downloaded
                    * Add final status of models
                </p>
            </b-tab>
            <b-tab title="About">
                <p>
             The Visual Document Index (VDI) browser application is designed to solve two problems:
             <ul>
                <li>flexibly search large amounts of text across multiple documents</li>
                <li>select, organize, and output references for follow-on work</li>
             </ul>
             It addresses these problems fairly well, but with some caveats.  Please review 
             the <code>Add More Files</code> modal for details.  
             
             These caveats will be addressed as the project progresses.
                </p>

            </b-tab>
            <b-tab title="Disabled" disabled><p>I'm a disabled tab!</p></b-tab>
          </b-tabs>
        </div>

     </b-modal>
 </template>


<script>

export default {
    name: "About"
}

</script>


<style scoped>
.fixed-large{
    width: 150px !important;
    margin: 5px;
}
</style>