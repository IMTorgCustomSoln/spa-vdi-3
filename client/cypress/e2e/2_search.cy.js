describe('search across files', () => {
    it('search Fuzzy with upload files', () => {
      cy.visit('/')
      cy.get('#btnImport').click();
      const fileName = ['./cypress/fixtures/10469527483063392000-cs_nlp_2301.09640.pdf','./cypress/fixtures/51172457010912950000-prob_2301.09751.pdf']
      cy.get('div > .custom-file-upload').click().selectFile(fileName);
      cy.get('.btn-success').click();
      cy.get('.btn-success').click();
      cy.get('#search-field').type('compute');
      cy.get('#results-summary > :nth-child(1) > div').should($el =>{
        expect($el[0].innerText).to.contain('Search returned 31 hits, in 2 documents\nTerms used: comput')
      })
    }),
    it('search Exact with upload files', () => {
      cy.visit('/')
      cy.get('#btnImport').click();
      const fileName = ['./cypress/fixtures/10469527483063392000-cs_nlp_2301.09640.pdf','./cypress/fixtures/51172457010912950000-prob_2301.09751.pdf']
      cy.get('div > .custom-file-upload').click().selectFile(fileName);
      cy.get('.btn-success').click();
      cy.get('.btn-success').click();
      cy.get('#__BVID__115__BV_toggle_').click();
      cy.get(':nth-child(2) > .dropdown-item').click();
      cy.get('#search-field').clear('compute');
      cy.get('#search-field').type('compute');
      cy.get('#results-summary > :nth-child(1) > div').should($el =>{
        expect($el[0].innerText).to.contain('Search returned 4 hits, in 1 documents\nTerms used: compute')
      })
    }),
    it.only('read snippets search Exact with upload files', () => {
      cy.visit('/')
      cy.get('#btnImport').click();
      const fileName = ['./cypress/fixtures/10469527483063392000-cs_nlp_2301.09640.pdf','./cypress/fixtures/51172457010912950000-prob_2301.09751.pdf']
      cy.get('div > .custom-file-upload').click().selectFile(fileName);
      cy.get('.btn-success').click();
      cy.get('.btn-success').click();
      cy.get('#__BVID__115__BV_toggle_').click();
      cy.get(':nth-child(2) > .dropdown-item').click();
      cy.get('#search-field').clear('compute');
      cy.get('#search-field').type('compute');
      cy.get('#results-summary > :nth-child(1) > div').should($el =>{
        expect($el[0].innerText).to.contain('Search returned 4 hits, in 1 documents\nTerms used: compute')
      })
      //TODO: what to do to test snippets
      cy.get('#btn-radios-1 > :nth-child(2) > span').click();
      cy.get('#btn-radios-1_BV_option_1').check();
      cy.get('#__BVID__135__row_1 > [aria-colindex="2"]').click();
      cy.get(':nth-child(3) > .snippet > div').click();
      cy.get(':nth-child(4) > .snippet > div').click();
      cy.get(':nth-child(5) > .snippet > div').click();
      cy.get(':nth-child(6) > .snippet > div').click();
    }),
    it('search Models with backend workspace ', () => {
      cy.visit('/')
      cy.get('#btnImport').click();
      cy.get('#tabWorkspace___BV_tab_button__').click();
      const fileName = ['./cypress/fixtures/asr-VDI_ApplicationStateData_v0.2.1_20240710_0-6.gz']
      cy.get('form > .custom-file-upload').click().selectFile(fileName);
      cy.get('#import-modal___BV_modal_footer_ > div > .btn').click();
      cy.get('#__BVID__117__BV_toggle_').click();
      cy.get(':nth-child(4) > .dropdown-item').click();
      cy.get('#results-summary > :nth-child(1) > div').should($el =>{
        expect($el[0].innerText).to.contain('Search returned 18 hits, in 6 documents\nTerms used: score')
      })
    })
  })