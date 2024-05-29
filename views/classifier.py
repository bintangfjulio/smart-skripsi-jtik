from flask import request, Blueprint, jsonify, current_app
from middleware import role_required
from models.lecturer import Lecturer


classifier = Blueprint('classifier', __name__, template_folder='templates', url_prefix='/dashboard/classifier')

@classifier.route('/inference', methods=['POST'])
@role_required('pengguna')
def inference():
    abstrak = request.get_json().get('abstrak')
    kata_kunci = request.get_json().get('kata_kunci')

    try:
        probs, kbk = current_app.inference.classification(abstrak, kata_kunci)
        lecturers = Lecturer.fetch(kelompok_bidang_keahlian=kbk)
        return jsonify(message={'probs': probs, 'lecturers': lecturers}, status="success"), 200
    
    except Exception as e:
        return jsonify(message={'error': 'Server error'}, status="error"), 500