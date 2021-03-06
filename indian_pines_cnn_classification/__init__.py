# -*- coding: utf-8 -*-

"""Top-level package for Indian Pines CNN Classification."""

__author__ = """Kitware Inc"""
__email__ = 'kitware@kitware.com'
__version__ = '0.0.0'


from girder_worker import GirderWorkerPluginABC


class IndianPinesCNNClassifciation(GirderWorkerPluginABC):
    def __init__(self, app, *args, **kwargs):
        self.app = app

    def task_imports(self):
        # Return a list of python importable paths to the
        # plugin's path directory
        return ['indian_pines_cnn_classification.create',
                'indian_pines_cnn_classification.train',
                'indian_pines_cnn_classification.validate']
