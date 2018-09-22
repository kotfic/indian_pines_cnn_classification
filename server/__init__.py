from __future__ import print_function

import json
import six

from girder import events
from girder.api import access
from girder.api.describe import autoDescribeRoute, Description
from girder.api.rest import ensureTokenScopes, filtermodel, Resource
from girder.models.item import Item
from girder.models.folder import Folder
from girder.plugins.jobs.models.job import Job
from girder.constants import AccessType
from girder_worker_utils.transforms.girder_io import GirderUploadToItem
from indian_pines_cnn_classification.create import preprocess_data as _preprocess_data
from indian_pines_cnn_classification.train import train_model as _train_model
from indian_pines_cnn_classification.validate import classify as _classify
from indian_pines_cnn_classification.transforms import ItemFilesToFolder, FolderFilesToItem



class IndianPines(Resource):
    def __init__(self):
        super(IndianPines, self).__init__()

        self.resourceName = 'indian_pines'

        self.route('POST', ('preprocess_data', ), self.preprocess_data)
        self.route('POST', ('train_model', ), self.train_model)
        self.route('POST', ('classify', ), self.classify)

    @filtermodel(model=Job)
    @access.public
    @autoDescribeRoute(
        Description('Preprocess Data')
        .modelParam('data_path', '', required=True, model=Item,
                    level=AccessType.READ, paramType='query', destName='data_path')
        .modelParam('output_path', '', required=True, model=Item,
                    level=AccessType.WRITE, paramType='query', destName='output_path')
        .param('numComponents', '', required=False, dataType='int', default=30)
        .param('windowSize', '', required=False, dataType='int', default=5)
        .param('testRatio', '', required=False, dataType='float', default=0.25))
    def preprocess_data(self, data_path, output_path, numComponents, windowSize, testRatio):

        async_result = _preprocess_data.delay(

            data_path=ItemFilesToFolder(data_path['_id']),
            output_path='/tmp/{}/'.format(output_path['_id']),
            numComponents=numComponents,
            windowSize=windowSize,
            testRatio=testRatio,

            girder_result_hooks=[
                FolderFilesToItem(output_path['_id'])
            ])

        return async_result.job


    @filtermodel(model=Job)
    @access.public
    @autoDescribeRoute(
        Description('Preprocess Data')
        .modelParam('data_path', '', required=True, model=Item,
                    level=AccessType.READ, paramType='query', destName='data_path')
        .modelParam('model_path', '', required=True, model=Item,
                    level=AccessType.WRITE, paramType='query', destName='model_path')
        .param('numComponents', '', required=False, dataType='int', default=30)
        .param('windowSize', '', required=False, dataType='int', default=5)
        .param('testRatio', '', required=False, dataType='float', default=0.25))
    def train_model(self, data_path, model_path, numComponents, windowSize, testRatio):

        async_result = _train_model.delay(
            data_path=ItemFilesToFolder(str(data_path['_id'])),
            model_path='/tmp/{}.h5'.format(model_path['_id']),
            numPCAcomponents=numComponents,
            windowSize=windowSize,
            testRatio=testRatio,
            girder_result_hooks=[
                GirderUploadToItem(
                    str(model_path['_id']), upload_kwargs={'filename': 'model.h5'})
            ]
        )

        return async_result.job


    @filtermodel(model=Job)
    @access.public
    @autoDescribeRoute(
        Description('Preprocess Data')
        .modelParam('data_path', '', required=True, model=Item,
                    level=AccessType.READ, paramType='query', destName='data_path')
        .modelParam('output_path', '', required=True, model=Item,
                    level=AccessType.WRITE, paramType='query', destName='output_path')
        .param('numComponents', '', required=False, dataType='int', default=30)
        .param('windowSize', '', required=False, dataType='int', default=5)
        .param('testRatio', '', required=False, dataType='float', default=0.25))
    def classify(self, data_path, output_path, numComponents, windowSize, testRatio):
        pass
#         async_result = _preprocess_data.delay(
#
#             data_path=ItemFilesToFolder(data_path['_id']),
#             output_path='/tmp/{}/'.format(output_path['_id']),
#             numComponents=numComponents,
#             windowSize=windowSize,
#             testRatio=testRatio,
#
#             girder_result_hooks=[
#                 FolderFilesToItem(output_path['_id'])
#             ])
#
#         return async_result.job



def load(info):
    """
    Load the plugin into Girder.

    :param info: a dictionary of plugin information.  The name key contains the
                 name of the plugin according to Girder.
    """
    info['apiRoot'].indian_pines = IndianPines()
