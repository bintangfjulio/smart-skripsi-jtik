import os
import middleware

from dotenv import load_dotenv
from flask import Flask, redirect, render_template, url_for
from views import views
from flask_login import current_user
from inference_config import Inference


load_dotenv(override=True)

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.getenv('SECRET_KEY')

middleware.init_middleware(app)
app.inference = Inference()

for view in views:
    app.register_blueprint(view)


@app.route('/')
def index():
    if current_user.is_authenticated:
        if current_user.role == 'admin':
            return redirect(url_for('dashboard.lecturer'))
        
        elif current_user.role == "pengguna":
            return redirect(url_for('dashboard.classifier'))
        
    return redirect(url_for('auth.sign_in'))


@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', message="Page Not Found", code=404), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return render_template('error.html', message="Request Not Allowed", code=405), 405


@app.route('/unauthorized')
def unauthorized():
    return render_template('error.html', message="Forbidden Access", code=403), 403


if __name__ == '__main__':
    app.run(debug=True)