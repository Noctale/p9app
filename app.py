
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json

DATA_URL = ('https://p9test.azurewebsites.net/api/HttpExemple?user=')

def functions_request(user_id):
  response = requests.get(DATA_URL+str(user_id))

  if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

  decoded = json.loads(response.content)
  reco = pd.DataFrame(decoded)#.set_index('article_id')
  return reco


st.title('Content Recommendations')
user_id = st.number_input("user_id :", value = 0, min_value = 0)
st.write(functions_request(user_id))


