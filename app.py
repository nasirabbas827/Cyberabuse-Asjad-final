from flask import Flask, request, render_template
import joblib
import re

app = Flask(__name__)

# Load models (must match the .joblib filenames you saved earlier)
models = {
    'Logistic Regression': joblib.load('logistic_regression.joblib'),
    'Decision Tree': joblib.load('decision_tree.joblib'),
    'Random Forest': joblib.load('random_forest.joblib')
}
tfidf = joblib.load('tfidf_vectorizer.joblib')

# Sample Roman Urdu stop words
stop_words = {'hai', 'aur', 'ke', 'ka', 'ki', 'ko', 'mein', 'se', 'bhi', 'ye', 'wo', 'nahi'}

# Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.strip().split()
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    comment = ''
    selected_model = 'Logistic Regression'  # Default model
    if request.method == 'POST':
        comment = request.form.get('comment', '')
        selected_model = request.form.get('model', 'Logistic Regression')
        if comment:
            cleaned_input = preprocess_text(comment)
            vectorized_input = tfidf.transform([cleaned_input])
            pred = models[selected_model].predict(vectorized_input)[0]
            prediction = 'Non-Abusive' if pred == 'non-abusive' else 'Abusive'
    return render_template(
        'index.html',
        prediction=prediction,
        comment=comment,
        selected_model=selected_model,
        models=models.keys()
    )

if __name__ == '__main__':
    app.run(debug=True)
