import pandas as pd
import flask
from sentence_transformers import SentenceTransformer, util
from flask import Flask, request, jsonify, render_template, url_for
import json
import os
from dotenv import load_dotenv
from json import loads, dumps
from elasticsearch import Elasticsearch

load_dotenv()

ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL")
ELASTICSEARCH_USERNAME = os.getenv("ELASTICSEARCH_USERNAME")
ELASTICSEARCH_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD")

INDEX_NAME = "list"
CSV_FILE_PATH = "Noise.csv"
data = pd.read_csv("NOISE.csv")
data = data.fillna(0)
cols = ['a-link-normal href', 's-image src', 's-label-popover href',
       'a-color-secondary', 'a-color-base', 'a-popover-preload',
       'a-link-normal href 2', 'a-link-normal', 'a-link-normal href 3', 'a-popover-trigger href', 'a-link-normal href 4', 'a-size-base', 'a-size-base href', 'a-offscreen', 'a-price-symbol',  'a-row', 'a-row 2', 'a-row 3', 'a-badge-text']

data = data.drop(columns= cols)

data =data.rename(columns= {'a-size-medium': 'title',
                      'a-icon-alt': "rating",
                      'a-price-whole': 'current-price',
                      'a-offscreen 2': 'old-price',
                      'a-text-bold': 'delivery-by' } )

def upload_csv_to_elasticsearch():
    # Load the SentenceTransformer model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Connect to Elasticsearch
    es = Elasticsearch(
        ELASTICSEARCH_URL,
        basic_auth=(ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD)
    )

    # Read the CSV file
    data = pd.read_csv(CSV_FILE_PATH, encoding="ISO-8859-1")

    for idx, row in data.iterrows():
        title = row["Title"]
        embedding = model.encode(title).tolist()

        doc = {
            "Title": title,
            "Rating": row["Rating"],
            "Price after Discount": row["Price after Discount"],
            "MRP": row["MRP"],
            "Delivery By": row["Delivery By"],
            "embedding": embedding
        }

        es.index(index=INDEX_NAME, body=doc)

    print("✅ All documents uploaded successfully!")




#attributes = ["title", "rating", "delivery-by"]
#for attr in attributes:
#  productEmbeddings = model.encode(data['attr'].astype(str).tolist(),  convert_to_tensor=True)
#productEmbeddings= {}
productEmbeddings = model.encode(data['title'].astype(str).tolist(),  convert_to_tensor=True)


app = Flask(__name__)

@app.route("/ping")
def ping():
    if es.ping():  
        return jsonify({"status": "Elasticsearch is up!"})
    else:
        return jsonify({"error": "Elasticsearch is down!"}), 500

@app.route('/')
def fun():
   return render_template("index.html")


@app.route('/predict', methods=['GET', 'POST'])
def recommend():  
  userInput = request.form['input']

  es_results = es.search(
        index="products",
        body={
            "query": {
                "match": {
                    "title": userInput
                }
            }
        }
    )
  userEmbeddings = model.encode(userInput, convert_to_tensor=True)
  similarities_pct = util.pytorch_cos_sim(userEmbeddings, productEmbeddings)[0] *100
  data['similarity'] = similarities_pct
  #result = dumps(result, indent=4)
  top_5 = data.nlargest(5, 'similarity')
  enhanced_results = []
  for _, row in top_5.iterrows():
      es_data = es.get(index="products", id=row.name)['_source']
      enhanced_results.append(es_data)
    
        
  return render_template('predict.html', result=enhanced_results)
  

if __name__ == '__main__':
    create_index()
    upload_csv_to_elasticsearch()
    app.run(debug=True)
