import os.path
import tensorflow

from lesa.models.preprocessing import get_tiles, model_out_to_png
from flask import Flask, render_template, request, send_file


app = Flask(__name__)


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/tile', methods=['GET'])
def get_tile():
    file_name = f"{request.args['z']}_{request.args['y']}_{request.args['x']}.png"

    if os.path.exists(os.path.join('static', 'tiles', file_name)):
        return send_file(file_name, mimetype='image/png')
    else:
        return send_file("static/empty_tile.png", mimetype='image/png')


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()

    try:
        tiles = get_tiles(data['tiles_coords'])
        raw_masks = model.predict(tiles)
        pngs = model_out_to_png(raw_masks)

        for k, v in zip(data['tiles_coords'], pngs):
            v.save(os.path.join('static', 'tile', f"{k['z']}_{k['y']}_{k['x']}.png"))

        return 'Ok', 200
    except:
        return "Not Found", 404


def run_server():
    app.run(port=80)


if __name__ == "__main__":
    global model
    model = tensorflow.keras.models.load_model("../../tests/lesa_model_best.h5")

    run_server()
