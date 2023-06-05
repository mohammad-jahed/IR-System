# This is a sample Python script.
import pickle

import nltk
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from flask import Flask, request, jsonify

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


import IR1
import bundle
import test
import tesapi

app = Flask(__name__)
# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     df = IR1.read_df(path='dataset/docs.csv')
#     tokenized_df = IR1.tokenize_df(df.head(10))
#     print(f'tokenized_df: {tokenized_df}')
#     clear_data = IR1.clean_data(tokenized_df)
#     print(f'clear_data: {clear_data}')
#     merged_text = [' '.join(row) for row in clear_data]
#     tfidf_vectors, features = IR1.tfidf_vector(merged_text)
#     print(f'tfidf_vectors {tfidf_vectors}\nfeatures: {features}')

# if __name__ == '__main__':
#     df = pd.read_csv('dataset/docs.csv')  # read data
#     #df = df.head(100)
#     df['processed_text'] = df['text'].apply(bundle.preprocess_text)  # Data Preprocessing
#     document_vectors = bundle.represent_data(df['processed_text'])  # Data Representation
#     inverse_index = bundle.build_index(df['processed_text'])  # Indexing
#     # match query
#     queries = pd.read_csv('dataset/queries.csv')#.head(10)  # Read queries
#     result_matches = bundle.process_queries(queries, document_vectors, inverse_index, df)  # mathing and ranking queries
#     print(result_matches)
#     #######################################################################################
#     qrels = pd.read_csv('dataset/qrels.csv')
#     relevant_docs = qrels.loc[qrels['query_id'] == queries['query_id'][0], 'doc_id']
#     print(relevant_docs)
#
#     intersection = np.intersect1d(relevant_docs, result_matches[queries['query_id'][0]])
#     print(intersection)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


# if __name__ == '__main__':
#     app.run(debug=True)
#     df = pd.read_csv("dataset/docs.csv")
#
#     # df = df.head(100000)
#
#     df['processed_text'] = df['text'].apply(test.preprocess_text)
#
#     document_vectors = test.represent_text(df['processed_text'])
#
#     inverse_index = test.build_index(df['processed_text'])
#     #
#     # with open('vectorized.pickle', 'wb') as file:
#     #     pickle.dump(document_vectors, file)
#     # with open('inverse_index.pickle', 'wb') as file:
#     #     pickle.dump(inverse_index, file)
#
#     queries = pd.read_csv('dataset/queries.csv')#.head(5)  # Read queries
#
#     # with open('document_vectors.pickle', 'rb') as f:
#     #     document_vectors1 = pickle.load(f)
#     # with open('inverse_index.pickle', 'rb') as f:
#     #     inverse_index1 = pickle.load(f)
#
#     result_matches = test.process_queries(queries, document_vectors, inverse_index, df)
#     print(result_matches)
#
#
#     @app.route('/process_queries', methods=['POST'])
#     def process_queries():
#         query_data = request.get_json()
#         queries_api = query_data['queries']
#         # document_vectors = query_data['document_vectors']
#         # inverse_index = query_data['inverse_index']
#         # df = query_data['df']
#
#         match_result = {}
#         for index, query in queries_api.iterrows():
#             processed_query = test.preprocess_text(queries_api)
#             candidate_docs = test.get_candidate_docs(processed_query, inverse_index)
#             query_results = test.match_query(processed_query, document_vectors, candidate_docs, df)
#             match_result[query['query_id']] = [doc['doc_id'] for doc in query_results]
#
#         return jsonify(match_result)
#
#
#     qrels = pd.read_csv('dataset/qrels.csv')
#
#     relevant_docs = qrels.loc[qrels['query_id'] == queries['query_id'][0], 'doc_id']
#     print(relevant_docs)
#
#     #Compare
#     intersection = np.intersect1d(relevant_docs, result_matches[queries['query_id'][0]])
#     print(intersection)
#
#     evaluation_results = {}
#     test.evaluation(queries, result_matches, qrels, evaluation_results)
#     print(evaluation_results)

if __name__ == '__main__':
    # app.run(debug=True)
    df = pd.read_csv("dataset/docs.csv")

    # df = df.head(100000)

    df['processed_text'] = df['text'].apply(tesapi.preprocess_text)

    document_vectors = tesapi.represent_text(df['processed_text'])

    inverse_index = tesapi.build_index(df['processed_text'])

    queries = pd.read_csv('dataset/queries.csv')  # Read queries

    qrels = pd.read_csv('dataset/qrels.csv')


    def process_queries(queries, document_vectors, inverse_index, df):
        match_result = {}
        for (index, query_id, text) in queries.values:
            top_k = len(qrels.loc[qrels['query_id'] == query_id, 'doc_id'])
            processed_query = tesapi.preprocess_text(text)
            candidate_docs = tesapi.get_candidate_docs(processed_query, inverse_index)
            query_results = tesapi.match_query(processed_query, document_vectors, candidate_docs, df, top_k)
            match_result[query_id] = [doc['doc_id'] for doc in query_results]
        return match_result


    result_matches = process_queries(queries, document_vectors, inverse_index, df)
    print(result_matches)

    evaluation_results = {}
    eval = tesapi.evaluation(queries, result_matches, qrels, evaluation_results)
    print(eval)


    @app.route('/process_queries', methods=['POST'])
    def process_queries():
        query_data = request.get_json()
        query = query_data['text']
        id = queries.loc[queries['text'] == query, 'query_id']
        top_k = len(qrels.loc[qrels['query_id'] == id, 'doc_id'])
        match_result = {}
        processed_query = tesapi.preprocess_text(query)
        candidate_docs = tesapi.get_candidate_docs(processed_query, inverse_index)
        query_results = tesapi.match_query(processed_query, document_vectors, candidate_docs, df, top_k)
        match_result[id] = [doc['doc_id'] for doc in query_results]
        return jsonify(match_result)
