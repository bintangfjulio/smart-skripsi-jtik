import os

from dotenv import load_dotenv
from flask import Flask, redirect
from views import views


load_dotenv(override=True)

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.getenv('SECRET_KEY')

for view in views:
    app.register_blueprint(view)

@app.route('/')
def index():
    return redirect('/sign-in')

if __name__ == '__main__':
    app.run(debug=True)