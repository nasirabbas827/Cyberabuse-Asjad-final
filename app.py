from flask import Flask, request, render_template
import joblib
import re

app = Flask(__name__)

# ============================================
#  Load Trained Models and TF-IDF Vectorizer
# ============================================

models = {
    'Logistic Regression': joblib.load('logistic_regression_model.joblib'),
    'Decision Tree': joblib.load('decision_tree_model.joblib'),
    'Random Forest': joblib.load('random_forest_model.joblib'),
    'SVM (Linear)': joblib.load('svm_(linear)_model.joblib')
}

vectorizer = joblib.load('tfidf_vectorizer.joblib')

# ============================================
#  Roman Urdu Stopwords (Extended List)
# ============================================

stop_words = {
    'hai', 'aur', 'ke', 'ka', 'ki', 'ko', 'mein', 'se', 'bhi', 'ye', 'wo',
    'nahi', 'kya', 'tum', 'main', 'mera', 'teri', 'tera', 'hum', 'un', 'apna',
    'tha', 'thi', 'hain', 'par', 'ab', 'jab', 'tak', 'chal', 'raha', 'rahi',
    'to', 'lekin', 'aur', 'is', 'us', 'kaun', 'kyun', 'magar', 'agar'
}

# ============================================
#  Text Preprocessing Function
# ============================================

def preprocess_text(text):
    text = text.lower()  # Lowercase conversion
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove punctuation/symbols
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    tokens = text.split()
    tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords
    return ' '.join(tokens)

# ============================================
#  Flask Routes
# ============================================

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    comment = ''
    selected_model = 'Logistic Regression'  # Default model

    if request.method == 'POST':
        comment = request.form.get('comment', '')
        selected_model = request.form.get('model', 'Logistic Regression')

        if comment:
            cleaned_text = preprocess_text(comment)
            transformed_text = vectorizer.transform([cleaned_text])
            pred = models[selected_model].predict(transformed_text)[0]

            # Map label to human-readable form
            if pred in ['A', 1]:
                prediction = 'Abusive'
            elif pred in ['NA', 0]:
                prediction = 'Non-Abusive'
            else:
                prediction = f'Predicted Label: {pred}'

    return render_template(
        'index.html',
        prediction=prediction,
        comment=comment,
        selected_model=selected_model,
        models=models.keys()
    )

# ============================================
#  Run the Flask App
# ============================================

if __name__ == '__main__':
    app.run(debug=True)
