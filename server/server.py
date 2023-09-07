import tensorflow

from flask import Flask, render_template, request, send_file, jsonify
from lesa.server.tile_processing import preprocess_tiles, postprocess_tiles, load_tiles


app = Flask(__name__)


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()

    try:
        tiles, width, height = load_tiles(data["tiles_coords"])

        input_batch = preprocess_tiles(tiles, width, height)
        out_batch = model.predict(input_batch)

        json = postprocess_tiles(
            out_batch=out_batch,
            width=width,
            height=height,
            start_tile_coords=data["tiles_coords"][0],
            analyze_area_polygon_dots=data["analyze_area_polygon"]
        )

        return jsonify(json)
    except Exception as ex:
        print(ex)

        return 'Not Found', 404


def run_server(port: int, availible_models: list[str]):
    global models



    app.run(port=port)
