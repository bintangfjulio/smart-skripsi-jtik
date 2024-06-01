from flask import request, Blueprint, jsonify, current_app
from middleware import role_required
from models.lecturer import Lecturer
from models.history import History
from datetime import datetime
from flask_login import current_user


classifier = Blueprint('classifier', __name__, template_folder='templates', url_prefix='/dashboard/classifier')

@classifier.route('/inference', methods=['POST'])
@role_required('pengguna')
def inference():
    abstrak = request.get_json().get('abstrak')
    kata_kunci = request.get_json().get('kata_kunci')

    try:
        probs, kbk = current_app.inference.classification(abstrak, kata_kunci)
        lecturers = Lecturer.fetch(kelompok_bidang_keahlian=kbk)

        history = History(abstrak=abstrak, kata_kunci=kata_kunci, probabilitas=probs, kelompok_bidang_keahlian=kbk, tanggal_inferensi=datetime.now())
        history.save(current_user.id)

        return jsonify(message={'probs': probs, 'lecturers': lecturers, 'kbk': kbk}, status="success"), 200
    
    except Exception as e:
        return jsonify(message={'error': 'Server error'}, status="error"), 500