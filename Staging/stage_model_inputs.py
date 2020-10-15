import os
import boto3

from utilities import report
from paths import input_dir, sam_root

session = boto3.session.Session(profile_name='sam')
c = session.get_credentials()
s3 = session.resource('s3')
sam_staged_bucket = s3.Bucket('sam-staged-inputs')

def upload_file(local, remote):
    report("{} -> {}".format(local, remote))
    try:
        sam_staged_bucket.upload_file(local, remote)
    except Exception as e:
        raise e


def main():
    for a, _, c in os.walk(input_dir):
        for f in c:
            d = os.path.join(a, f)
            new_d = d.lstrip(sam_root).replace(os.path.sep, os.path.altsep)
            upload_file(d, new_d)


main()
