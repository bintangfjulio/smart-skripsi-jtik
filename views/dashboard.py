from flask import Blueprint, render_template
from middleware import role_required


dashboard = Blueprint('dashboard', __name__, template_folder='templates', url_prefix='/dashboard')

@dashboard.route('/lecturer', methods=['GET'])
@role_required('admin')
def lecturer():
    return render_template('dashboard/lecturer.html')

@dashboard.route('/classifier', methods=['GET'])
@role_required('user')
def classifier():
    return render_template('dashboard/classifier.html')