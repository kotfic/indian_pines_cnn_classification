from __future__ import print_function
import tempfile
import os
import shutil

from girder_worker_utils.transforms.girder_io import GirderClientTransform, GirderClientResultTransform

class ItemFilesToFolder(GirderClientTransform):
    def _repr_model_(self):
        return "{}('{}')".format(self.__class__.__name__, self.item_id)

    def __init__(self, item_id, **kwargs):
        super(ItemFilesToFolder, self).__init__(**kwargs)
        self.item_id = str(item_id)

    def transform(self):
        self.folder_path = tempfile.mkdtemp()
        self.gc.downloadItem(self.item_id, self.folder_path, name='transform')
        return os.path.join(self.folder_path, 'transform')

    def cleanup(self):
        shutil.rmtree(self.folder_path)


class FolderFilesToItem(GirderClientResultTransform):
    def _repr_model_(self):
        return "{}('{}')".format(self.__class__.__name__, self.item_id)

    def __init__(self, item_id, **kwargs):
        super(FolderFilesToItem, self).__init__(**kwargs)
        self.item_id = str(item_id)

    def transform(self, path):
        print(path)
        self.path = path
        for f in os.listdir(path):
            print("Uploading: {}".format(os.path.join(path, f)))
            self.gc.uploadFileToItem(self.item_id, os.path.join(path, f))

        return self.item_id

    def cleanup(self):
        shutil.rmtree(self.path)
