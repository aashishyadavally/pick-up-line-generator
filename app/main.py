# Run by typing python3 main.py

# Import basics
import re
import os
import pickle

# Import stuff for our web server
from flask import Flask, flash, request, redirect, url_for, render_template
from flask import send_from_directory
from flask import jsonify
from utils import get_base_url, allowed_file, and_syntax

# Import stuff for text pre-processing and models
import numpy as np
import nltk
nltk.download('punkt')
import torch
from aitextgen import aitextgen
from gensim.models import Word2Vec


# Load up the models into memory
ai = aitextgen(to_gpu=False, model_folder="models/trained_model_gpt2")
rf_model = pickle.load(open('models/random_forest_model_avg.pkl', 'rb'))
w2v_model = Word2Vec.load('models/w2v.bin')

NON_ALPHANUM = re.compile(r'[\W]')
NON_ASCII = re.compile(r'[^a-z0-1\s]')

# Setup the webserver
# Port may need to be changed if there are multiple flask servers running on same server
#port = 12346
#base_url = get_base_url(port)
#app = Flask(__name__, static_url_path=base_url+'static')


# Deployment code - uncomment the following line of code when ready for production
app = Flask(__name__)



def sent_vectorizer(sent):
    """Takes in a sentence and returns the average word2vec embedding of all words
    in this sentence that are in the vocab of the model.

    Inputs:
    -------
        sent (str):
            a string of a sentence to embedd
        model (gensim.models.Word2Vec):
            an already trained Word2Vec model

    Output:
    -------
        avgembedding (np.ndarray):
            A 100-dimension long numpy vector of the average Word2vec embedding of all
            the words of ``sent`` that appear in the vocabulary of Word2Vec model.
    """
    sent_vec = np.zeros(100)
    numw = 0
    words = nltk.word_tokenize(sent)

    for w in words:
        if w in w2v_model.wv.index_to_key:
            sent_vec = np.add(sent_vec, w2v_model.wv[w])
            numw += 1
    avgembedding = sent_vec/numw
    return avgembedding


def clean_text(text):
    """Cleans text using regex.

    Arguments:
    ----------
        text (str):
            Text.
        no_non_ascii (str):
            Cleaned text.
    """
    lower = text.lower()
    no_punctuation = NON_ALPHANUM.sub(r' ', lower)
    no_non_ascii = NON_ASCII.sub(r'', no_punctuation)
    return no_non_ascii


@app.route('/')
#@app.route(base_url)
def home():
    return render_template('Home.html', generated=None)


@app.route('/', methods=['POST'])
#@app.route(base_url, methods=['POST'])
def home_post():
    return redirect(url_for('results'))


@app.route('/team')
#@app.route(base_url + '/team')
def team():
    return render_template('Team.html', generated=None)


@app.route('/results')
#@app.route(base_url + '/results')
def results():
    return render_template('Try-our-product.html', generated=None)


@app.route('/generate_text', methods=["POST"])
#@app.route(base_url + '/generate_text', methods=["POST"])
def generate_text():
    """
    View function that will return json response for generated text. 
    """
    prompt = request.form['prompt']
    if prompt is not None:
        prompt = str(prompt).strip()
        generated = ai.generate(
            n=2,
            batch_size=4,
            prompt=prompt,
            max_length=20,
            temperature=1.0,
            top_p=0.9,
            return_as_list=True
        )

    opinions = []
    for line in generated:
        cleaned_line = clean_text(line)
        embedding = sent_vectorizer(cleaned_line).reshape(-1, 100)
        opinion = rf_model.predict(embedding).item()
        if opinion == '1':
            opinions.append('<br><i> ( Meemaw <span style=\"color: #008000\">approves</span> this message! )</i>')
        elif opinion == '-1':
            opinions.append("<br><i> ( Meemaw <span style=\"color: #E53F2E\">doesn't approve</span> this message! )</i>")

    data = {'generated_ls': generated, 'opinions': opinions}
    return jsonify(data)


if __name__ == "__main__":
    '''
    coding center code
    '''
    # IMPORTANT: change the cocalcx.ai-camp.org to the site where you are editing this file.
    website_url = 'cocalc2.ai-camp.org'
    print(f"Try to open\n\n    https://{website_url}" + base_url + '\n\n')

    app.run(host = '0.0.0.0', port=port, debug=True)
    import sys; sys.exit(0)

    '''
    scaffold code
    '''
    # Only for debugging while developing
    # app.run(port=80, debug=True)
