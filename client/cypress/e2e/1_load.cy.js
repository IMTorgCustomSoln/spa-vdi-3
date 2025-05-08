describe('load files', () => {
  it('upload files', () => {
    cy.visit('/')
    cy.get('#btnImport').click();
    const fileName = ['./cypress/fixtures/10469527483063392000-cs_nlp_2301.09640.pdf','./cypress/fixtures/51172457010912950000-prob_2301.09751.pdf']
    cy.get('div > .custom-file-upload').click().selectFile(fileName);
    cy.get('.btn-success').click();
    cy.get('.btn-success').click();
  }),
  it('upload backend workspace', () => {
    cy.visit('/')
    cy.get('#btnImport').click();
    cy.get('#tabWorkspace___BV_tab_button__').click();
    const fileName = ['./cypress/fixtures/asr-VDI_ApplicationStateData_v0.2.1_20240710_0-6.gz']
    cy.get('form > .custom-file-upload').click().selectFile(fileName);
    cy.get('#import-modal___BV_modal_footer_ > div > .btn').click();
  })
  it('re-upload backend workspace with added separate document', () => {
    cy.visit('/')
    //workspace with models
    cy.get('#btnImport').click();
    cy.get('#tabWorkspace___BV_tab_button__').click();
    const fileName1 = ['./cypress/fixtures/asr-VDI_ApplicationStateData_v0.2.1_20240710_0-6.gz']
    cy.get('form > .custom-file-upload').click().selectFile(fileName1);
    cy.get('#import-modal___BV_modal_footer_ > div > .btn').click();
    //additional individual document
    cy.get('#btnImport').click();
    const fileName2 = ['./cypress/fixtures/10469527483063392000-cs_nlp_2301.09640.pdf']
    cy.get('div > .custom-file-upload').click().selectFile(fileName2);
    cy.get('.btn-success').click();
    cy.get('.btn-success').click();
    //search, take note, export workspace
    cy.get('#search-field').type('compute');
    cy.get('#btn-radios-1 > :nth-child(2)').click();
    cy.get('#btn-radios-1_BV_option_1').check();
    cy.get('#__BVID__137__row_6 > [aria-colindex="3"]').click();
    cy.get(':nth-child(3) > .snippet > div > [style="background-color: yellow"]').click();

    cy.get('[style="background-color: black;"] > .btn-group > :nth-child(3)', { timeout: 10000 }).click();
    cy.get('#btnSidebar').click();

    /*
    Save Workspace to `../fixtures/Test_1.3_v0.2.1.gz`
    */

    //upload new workspace
    cy.visit('/')
    cy.get('#btnImport').click();
    cy.get('#tabWorkspace___BV_tab_button__').click();
    const fileName3 = ['./cypress/fixtures/Test_1.3_v0.2.1.gz']
    cy.get('form > .custom-file-upload').click().selectFile(fileName3);
    cy.get('#import-modal___BV_modal_footer_ > div > .btn').click();
    cy.get('#btn-radios-1 > :nth-child(2)').click();
    cy.get('#btn-radios-1_BV_option_1').check();
    cy.get('#__BVID__137__row_6 > [aria-colindex="3"]').click();
    cy.get('.itemconfiguration > :nth-child(3)').click();
  })
})