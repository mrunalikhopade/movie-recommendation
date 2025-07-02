from flask import Flask, render_template, request
from recommender import get_recommendations

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = None
    if request.method == 'POST':
        movie = request.form.get('movie', '')
        recommendations = get_recommendations(movie)
    return render_template('index.html', recommendations=recommendations)

# We DO NOT include app.run() here; Render uses gunicorn.
