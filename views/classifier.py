import jsonify

from flask import Blueprint
from middleware import role_required


classifier = Blueprint('classifier', __name__, template_folder='templates', url_prefix='/dashboard/classifier')

@classifier.route('/inference', methods=['POST'])
@role_required('pengguna')
def inference():
    pass