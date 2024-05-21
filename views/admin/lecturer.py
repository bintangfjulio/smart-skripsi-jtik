from flask import Blueprint, render_template


lecturer = Blueprint('lecturer', __name__, template_folder='templates', url_prefix='/admin/lecturer')

@lecturer.route('/', methods=['GET'])
def lecturer():
    return render_template('admin/lecturer.html')