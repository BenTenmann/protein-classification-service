import tarfile

from flask import Flask, request, send_from_directory

from .__main__ import main

TARFILE_RESPONSE = 'model-weights.tar.gz'

app = Flask(__name__)


@app.route('/train', methods=['POST'])
def train_model():
    if request.is_json:
        manifest = request.get_json()
        output_path = main(manifest)  # runs training

        tar_file = output_path.parent / TARFILE_RESPONSE
        with tarfile.open(tar_file, 'w:gz') as file:
            file.add(output_path, arcname=output_path.name)
        return send_from_directory(tar_file.parent, tar_file.name, as_attachment=True), 200
    return {'error': 'Request must be JSON'}, 415
