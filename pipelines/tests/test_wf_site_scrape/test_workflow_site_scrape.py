#!/usr/bin/env python3
"""
Test site_scrape workflow
"""

__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "AGPL-3.0"

from workflows.workflow_site_scrape import (
    config,
    option1_tasks,
    option2_tasks,

    WorkflowSiteScrape
)

import pytest
import copy

config_op1 = copy.deepcopy(config)
config_op2 = copy.deepcopy(config)


@pytest.mark.skip(reason='training is too slow on local machine')
def test_prepare_models():
    workflow_site_scrape = WorkflowSiteScrape(config)
    check1 = workflow_site_scrape.prepare_models()
    assert check1 == True

def test_prepare_workspace():
    workflow_site_scrape = WorkflowSiteScrape(config)
    check1 = workflow_site_scrape.prepare_workspace()
    assert check1 == True

def test_run_option1_individual_pdf():
    config_op1['TASKS'].extend(option1_tasks)
    workflow_site_scrape_op1 = WorkflowSiteScrape(config_op1)

    check1 = workflow_site_scrape_op1.prepare_workspace()
    check2 = workflow_site_scrape_op1.run()
    assert check2 == True

def test_run_option2_grouped_url_docs():
    config_op2['TASKS'].extend(option2_tasks)
    workflow_site_scrape_op2 = WorkflowSiteScrape(config_op2)

    check1 = workflow_site_scrape_op2.prepare_workspace()
    check2 = workflow_site_scrape_op2.run()
    assert check2 == True