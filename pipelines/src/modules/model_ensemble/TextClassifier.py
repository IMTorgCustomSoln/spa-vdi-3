#!/usr/bin/env python3
"""
Classifier for topics of text models

"""

__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "AGPL-3.0"

from .Classifier import ClassifierTemplate
import concurrent.futures


class TextClassifier(ClassifierTemplate):
    """Text Classifier including an ensemble of model(s) and
    a coordinator of their results.
    
    
    """
    def __init__(self, name, config, models, coordinator):
        if name not in config['TRAINING_DATA_DIR'].keys():
            raise Exception(f'ERROR: the name {name} is not found in the `config["TRAINING_DATA_DIR"].keys()` ')
        super().__init__(name, config)
        self.models = models
        self.coordinator = coordinator

    def validate_models_input(self):
        """..."""
        checks = {}
        for model in self.models:
            check = model.validate(self.config)
            staged_result = model._get_staged_result()
            key = f"{staged_result['model_topic']}-{staged_result['topic_class']}"
            checks[key] = check
        return checks
    
    def finetune_models(self):
        """..."""
        checks = {}
        for model in self.models:
            if hasattr(model, 'finetune' ):
                check = model.finetune(self.config)
                staged_result = model._get_staged_result()
                key = f"{staged_result['model_topic']-{staged_result['topic_class']}}"
                checks[key] = check
        return checks

    def run(self, text):
        """Run inference on string of text.
        
        """
        workers = len(self.models)
        if workers==1:
            results = self.models[0].run(text)
            results = self.coordinator.run(results)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                futures = []
                for model in self.models:
                    futures.append( executor.submit(model.run, text) )
                done_and_not_done = self.coordinator.run(futures)
                results = done_and_not_done
        return results