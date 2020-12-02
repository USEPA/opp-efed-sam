from __future__ import absolute_import
import pymongo as pymongo
import json
import pandas as pd
import os

from ..paths import PathManager

paths = PathManager()

# from flask_qed.pram_flask.tasks import sam_status
# from pram_flask.tasks import sam_status
from celery_cgi import celery

IN_DOCKER = os.environ.get("IN_DOCKER")


# IN_DOCKER = "False"


class SamPostprocessor(object):

    def __init__(self, task_id):
        self.task_id = task_id
        self.sam_data = None
        self.huc8_summary = None
        self.huc12_summary = None
        self.status = celery.AsyncResult(self.task_id).status

    def connect_to_mongoDB(self):
        if IN_DOCKER == "False":
            # Dev env mongoDB
            mongo = pymongo.MongoClient(host='mongodb://localhost:27017/0')
            print("MONGODB: mongodb://localhost:27017/0")
        else:
            # Production env mongoDB
            mongo = pymongo.MongoClient(host='mongodb://mongodb:27017/0')
            print("MONGODB: mongodb://mongodb:27017/0")
        mongo_db = mongo['pram_tasks']
        mongo.pram_tasks.Collection.create_index([("date", pymongo.DESCENDING)], expireAfterSeconds=86400)
        # ALL entries into mongo.flask_hms must have datetime.utcnow() timestamp, which is used to delete the record after 86400
        # seconds, 24 hours.
        return mongo_db

    def get_sam_data(self):
        mongo_db = self.connect_to_mongoDB()
        posts = mongo_db.posts
        db_record = posts.find_one({'_id': self.task_id})
        print("Run status:" + self.status)
        data = json.loads(db_record["data"])
        self.sam_data = data
        return

    def calc_huc_summary(self):
        path_to_csv = paths.nhd_wbd_xwalk
        data = pd.DataFrame(self.sam_data['COMID']).T
        data.index = data.index.astype(str)
        huc_comid = pd.read_csv(path_to_csv, dtype=object)[['FEATUREID', 'HUC_12']]\
            .set_index('FEATUREID')
        print(12345, data.head())
        print(67899, huc_comid.head())
        data = data.join(huc_comid, how='inner')
        print(99999, data.head())
        print(999999, data.shape)
        data["HUC_8"] = data["HUC_12"].str.slice(0, 8)
        print(data.HUC_8.unique())
        print(data.HUC_12.unique())
        self.calc_huc8(data)
        self.calc_huc12(data)
        print(789, self.huc8_summary.head())
        print(101112, self.huc12_summary.head())
        return

    def calc_huc8(self, data):
        print("Post-processor: calculating HUC8s")
        try:
            huc8_summary = data.groupby('HUC_8').agg(['mean', 'max'])
            print(111, huc8_summary.columns)
            huc8_summary.columns = ["_".join(x) for x in huc8_summary.columns.ravel()]
            print(112, huc8_summary.columns)
            self.huc8_summary = huc8_summary
        except Exception as e:
            print("ERRROR A")
            print(e)
            self.huc8_summary = pd.DataFrame(columns=['HUC_8', 'acute_human_mean', 'acute_human_max'])
            return

    def calc_huc12(self, data):
        print("Post-processor: calculating HUC12s")
        try:
            huc12_summary = data.groupby('HUC_12').agg(['mean', 'max'])
            print(222, huc12_summary.columns)
            huc12_summary.columns = ["_".join(x) for x in huc12_summary.columns.ravel()]
            print(223, huc12_summary.columns)
            self.huc12_summary = huc12_summary
        except Exception as e:
            print("ERRROR B")
            print(e)
            self.huc12_summary = pd.DataFrame(columns=['HUC_12', 'acute_human_mean', 'acute_human_max'])
        return

    def append_sam_data(self):
        mongo_db = self.connect_to_mongoDB()
        posts = mongo_db.posts
        posts.update_one({'_id': self.task_id}, {'$set': {'huc8_summary': self.huc8_summary.to_json(orient='index'),
                                                          'huc12_summary': self.huc12_summary.to_json(orient='index')}})
        return


def replace_leading_0(huc_str):
    if len(huc_str) == 12:
        return huc_str
    elif len(huc_str) == 11:
        return '0' + huc_str
    else:
        raise NameError('Number that is neither 12 digits nor 11 digits!')
