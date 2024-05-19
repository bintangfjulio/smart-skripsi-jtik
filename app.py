import os

from dotenv import load_dotenv
from flask import Flask, redirect, render_template
from views import views


load_dotenv(override=True)

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.getenv('SECRET_KEY')

for view in views:
    app.register_blueprint(view)

@app.route('/')
def index():
    return redirect('/sign-in')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(debug=True)