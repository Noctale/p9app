import logging
from io import BytesIO
from azure.storage.blob import BlobServiceClient
import pandas as pd
import azure.functions as func
from p9class import *

connect_str = "DefaultEndpointsProtocol=https;AccountName=p9teststockage;AccountKey=AI0GHZZor7+Sg6MxDQvBoL1NtdEMhQA3mqxB98r+I3Oj6fZ/B3oeQ44uW/2SJxP/4XA4xfyqbs/X+ASti4jVPg==;EndpointSuffix=core.windows.net"
blob_service_client = BlobServiceClient.from_connection_string(connect_str)

upload_file_path = "P9_chaleyssin_nicolas/clicks"
blob_client_clicks = blob_service_client.get_blob_client(container="dftoblob", blob=upload_file_path)
with BytesIO() as input_blob:
    blob_client_clicks.download_blob().download_to_stream(input_blob)
    input_blob.seek(0)
    clicks = pd.read_csv(input_blob)


upload_file_path = "P9_chaleyssin_nicolas/articles"
blob_client_articles = blob_service_client.get_blob_client(container="dftoblob", blob=upload_file_path)
with BytesIO() as input_blob:
    blob_client_articles.download_blob().download_to_stream(input_blob)
    input_blob.seek(0)
    articles = pd.read_csv(input_blob)


# upload_file_path = "P9_chaleyssin_nicolas/duration"
# blob_client_duration = blob_service_client.get_blob_client(container="dftoblob", blob=upload_file_path)
# with BytesIO() as input_blob:
#     blob_client_duration.download_blob().download_to_stream(input_blob)
#     input_blob.seek(0)
#     cf_duration = pd.read_csv(input_blob)

data = {'clicks': clicks,
       'articles': articles}

cb_model_profile = ContentBased_mean(data)
#cf_model_duration = CollaborativeFiltering(cf_duration, data)
#hybrid_model = Hybrid(cb_model_profile, cf_model_duration, data)


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    user = req.params.get('user')
    if not user:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            user = req_body.get('user')

    if user:
        reco = cb_model_profile.recommend_items(int(user), nb = 5).to_json()
        return func.HttpResponse(reco)
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a user_id in the query string or in the request body for a personalized response.",
             status_code=200
        )
