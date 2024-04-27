from flask import Flask, request
from flask_cors import CORS
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import ktrain
import json
from newspaper import Article
import validators
app = Flask(__name__)
CORS(app)  # Enable CORS for all domains on all routes

# Define the paths to the trained ktrain predictor models
model_paths = {
    'fasttext': 'models/fastext',
    'bigru': 'models/bigru',
    'nbsvm': 'models/nbsvm'
}

def clean_texts(texts, max_length=350):
    cleaned_texts = []
    for text in texts:
            clean_text = re.sub(r"[^a-zA-Z0-9 čćđšžČĆĐŠŽ,.!?-]", "", text)
            clean_text = clean_text[:max_length]
            cleaned_texts.append(clean_text)
    return cleaned_texts

# Load predictors
def fetch_article_content(url):
    if not validators.url(url):
        return "Not a valid URL"
    try:
        article = Article(url)

        article.download()
        article.parse()
        texts=article.title +'.'+ article.text

        return (clean_texts(texts))
    except Exception as e:
        return "Not a good URL"


def get_prediction(p, text):
    # Depending on the model, we may get probabilities or just class predictions
    c = p.predict([text])[0]
    cat = {category: f"{probability * 100:.2f}%" for category, probability in c}
    jso = json.dumps(cat, indent=4)
    return (jso)


@app.route('/classify', methods=['GET'])
def classify():
    url = request.args.get('text', None)
    model = request.args.get('model', None)
    if model not in ('fasttext','bigru','nbsvm'):
        return json.dumps(
            {"error": "Model not specified or not found. Please specify 'fasttext', 'bigru', or 'nbsvm'."}), 404
    text=fetch_article_content(url)

    p = ktrain.load_predictor(model_paths[model])

    result = get_prediction(p, text)
    return (result)




if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')