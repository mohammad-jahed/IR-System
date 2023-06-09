from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from  dateutil.parser import parse
from datetime import datetime
from flask import Flask, request, jsonify
from gensim.corpora import Dictionary
from gensim.matutils import Sparse2Corpus
from gensim.models import TfidfModel ,Word2Vec
from nltk.probability import FreqDist

import string
import re
import contractions
import pandas as pd
import nltk as nk
import numpy as np
import re
import country_converter as coco
import pickle
import os

ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
vectorizer = TfidfVectorizer()
vec = CountVectorizer()
cc = coco.CountryConverter()

def handle_numbers(text):
    # Replace numbers with a special token
    text = re.sub("(?<![\d/-])\d+(?![\d/-])", "[NUMBER]", text)
    return text

def handle_countries(tokens):
  text = cc.pandas_convert(series = pd.Series(tokens, name='country'), to='ISO3', not_found=None)
  text = (' ').join(text)
  text = text.replace("united states", "USA")
  text = text.replace("united kingdom", "UK")
  return word_tokenize(text)

def handle_contractions(text):
    # Expand contractions in the text
    text = contractions.fix(text)
    return text

def remove_punctuation(text):
    # Define a translation table that excludes punctuation from dates
    date_punctuation = string.punctuation.replace('-', '')
    table = str.maketrans('', '', date_punctuation)

    # Split the text into words
    words = text.split()

    # Remove punctuation from each word, except for words that look like dates or contain special characters
    cleaned_words = []
    for word in words:
        if any(char.isdigit() for char in word) or any(char.isalpha() for char in word):
            # If the word contains alphanumeric characters, keep punctuation
            cleaned_word = word.translate(str.maketrans('', '', string.punctuation.replace('-', '')))
            cleaned_words.append(cleaned_word)
        else:
            # If the word contains only special characters or punctuation, remove it
            continue

    # Join the cleaned words back into a string
    cleaned_text = ' '.join(cleaned_words)
    return cleaned_text

def lowercase(text):
    # Convert the text to lowercase
    text = text.lower()
    return text

def handle_dates(text):
    # Define regular expression to match dates in text
    regex = r'\b\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\b'

    # Search for all dates in text
    dates = re.findall(regex, text)

    # Loop through all dates found and convert to the desired format
    for date in dates:
        # Try to parse the date - skip if it is an invalid date
        try:
            datetime_obj = datetime.strptime(date, '%d/%m/%Y')
        except ValueError:
            try:
                datetime_obj = datetime.strptime(date, '%Y/%m/%d')
            except ValueError:
                try:
                    datetime_obj = datetime.strptime(date, '%m/%d/%Y')
                except ValueError:
                    continue

        # Convert date to desired format
        formatted_date = datetime_obj.strftime('%Y-%m-%d')

        # Replace the original date in the text with the formatted date
        text = text.replace(date, formatted_date)

    # Return the modified text with handled dates
    return text

def lemmatize_with_pos(tokens):
    pos_tags = nk.pos_tag(tokens)
    pos_lemmas = []
    for word, pos in pos_tags:
        if pos.startswith('J'):
            # If the word is an adjective, use 'a' (adjective) as the POS tag
            pos_lemmas.append(lemmatizer.lemmatize(word, pos='a'))
        elif pos.startswith('V'):
            # If the word is a verb, use 'v' (verb) as the POS tag
            pos_lemmas.append(lemmatizer.lemmatize(word, pos='v'))
        elif pos.startswith('N'):
            # If the word is a noun, use 'n' (noun) as the POS tag
            pos_lemmas.append(lemmatizer.lemmatize(word, pos='n'))
        else:
            # For all other cases, use the default POS tag 'n' (noun)
            pos_lemmas.append(lemmatizer.lemmatize(word))
    return pos_lemmas


def preprocess_text(text_str):
    # text_str = str(text)
    # Normalization
    text_str = remove_punctuation(text_str)
    text_str = handle_dates(text_str)
    text_str = handle_numbers(text_str)
    text_str = handle_contractions(text_str)
    text_str = lowercase(text_str)

    # Tokenization
    tokens = word_tokenize(text_str)
    stop_words = set(stopwords.words('english'))
    # Removing stop worlds
    tokens = [token for token in tokens if token not in stop_words]

    # tokens = handle_countries(tokens)

    # Stemming
    tokens = [ps.stem(token) for token in tokens]

    # Lemmatization
    lemmatized_tokens = lemmatize_with_pos(tokens)
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text

# Data Representation
def represent_text(text):
    x = vectorizer.fit_transform(text)
    return x


#tf_idf
def docs_tf_tidf(document_vectors,path):
    # convert tf-idf vectors to gensim corpus
    corpus = Sparse2Corpus(document_vectors, documents_columns=False)
    # إنشاء نموذج tf-idf
    tfidf_model = TfidfModel(corpus)

    # تحويل التمثيلات إلى تمثيلات tf-idf gensim
    corpus_tfidf = tfidf_model[corpus]
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=4, compute_loss=True)
    save_model(path,model)
    return model

def save_model(model_path, model):
    if os.path.exists(model_path):
        os.remove(model_path)
    model.save(model_path)

def load_model(model_path):
    model = Word2Vec.load(model_path)
    return model

def query_procc(query):
    query_processed = preprocess_text(query)
    query_words = query_processed.split()
    print(query_processed)
    print(query_words)
    return query_words

def query_sss(query,model):
    query=query_procc(query)
    query_vector = np.zeros((100,))
    for word in query:
        if word in model.wv.key_to_index:
            query_vector += model.wv[word]
    query_vector /= len(query)
    similarities = []
    for i, text in enumerate(preprocessed_text):
        similarity = np.dot(query_vector,array[i]) / (np.linalg.norm(query_vector) * np.linalg.norm(array[i]))
        similarities.append((i, similarity))
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    recommended_texts = [df.iloc[i] for i, _ in similarities]
    print( recommended_texts)
    return recommended_texts

def complete_queriess(query,model):
    doc = nlp(query)
    # get the tokens
    tokens = [token.text for token in doc]
    vectors = []
    for token in tokens:
        if token in model.wv.key_to_index:
            vectors.append(model.wv[token])
    # combine the vectors to get the CBOW representation for the query
    cbow = sum(vectors) / len(vectors)

    # use the trained model to get the most similar words to the CBOW query
    similar_words = model.wv.most_similar(positive=[cbow])
    print(similar_words)

def process_queries(queries):
    match_result = {}
    # load model
    # model=load_model("my_model4")
    # Get tf_itf
    # tfidf_model = docs_tf_tidf(document_vectors)
    for (index, query_id, text) in queries.values:
        top_k = len(qrels.loc[qrels['query_id'] == query_id, 'doc_id'])
        print(text)
        # Rank the documents and retrieve the top results
        query_results = query_sss(text, model)
        # Store the results for the query
        match_result[query_id] = [doc['doc_id'] for doc in query_results]

    return match_result



import os
from gensim.models import Word2Vec


#Indexing
def build_index(data):
    inverse_index = {}
    for i, doc in enumerate(data):
        terms = doc.split(' ')
        for term in terms:
            if term not in inverse_index:
                inverse_index[term] = [i]
            else:
                inverse_index[term].append(i)
    return inverse_index

def match_query(query_vector, document_vectors, candidate_docs, data, top_k):
    query_vector = vectorizer.transform([query_vector])
    similarities = cosine_similarity(query_vector, document_vectors).flatten()

    # Rank the documents
    sorted_indices = similarities.argsort()[::-1]
    # print(sorted_indices)
    # print(data)
    ranked_documents = [data.iloc[i] for i in sorted_indices if i in candidate_docs]

    # Return the top-ranked documents
    search_results = ranked_documents[:top_k]
    return search_results

def get_candidate_docs(query_text, inverse_index):
    # Split the query text into terms
    query_terms = query_text.split(' ')

    # Retrieve the candidate document IDs for the query
    candidate_doc_ids = set(inverse_index.get(query_terms[0], []))
    for term in query_terms[1:]:
        candidate_doc_ids.intersection_update(set(inverse_index.get(term, [])))

    # Return the candidate document IDs
    return candidate_doc_ids


import pandas as pd


if __name__ == '__main__':
    df = pd.read_csv("dataset/docs.csv")

    preprocessed_text = df['text'].apply(preprocess_text)

    sentences = [str(sentence).split() for sentence in preprocessed_text]

    document_vectors = represent_text(preprocessed_text)

    model = docs_tf_tidf(document_vectors, "word2vec_model")

    model = Word2Vec.load('word2vec_model')

    array = []
    for i, text in enumerate(preprocessed_text):
        text_vector = np.zeros((100,))
        text_words = text.split()
        for word in text_words:
            if word in model.wv.key_to_index:
                text_vector += model.wv[word]
        text_vector /= len(text_words)
        array.append(text_vector)
    print(array[i])

    import spacy

    nlp = spacy.load('en_core_web_sm')

    queries = pd.read_csv("dataset/queries.csv")

    qrels = pd.read_csv("dataset/qrels.csv")

    result_matches = process_queries(queries)

    print(result_matches)

    query = "the game pits a group of armed humans against "
    complete_queriess(query, model)