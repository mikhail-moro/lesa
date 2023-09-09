from flask import Flask, render_template, request, jsonify
from .tile_processing import preprocess_tiles, postprocess_tiles, load_tiles


app = Flask(__name__)


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()

    tiles, width, height = load_tiles(data["tiles_coords"])

    input_batch = preprocess_tiles(tiles, width, height)
    out_batch = analyze_tiles(data["selected_model"], input_batch)

    if out_batch is not None:
        json = postprocess_tiles(
            out_batch=out_batch,
            width=width,
            height=height,
            start_tile_coords=data["tiles_coords"][0],
            analyze_area_polygon_dots=data["analyze_area_polygon"]
        )

        return jsonify(json)
    else:
        return jsonify({"polygons": None, "success": False, "message:": "Данная модель недоступна"})
    """
    try:
        tiles, width, height = load_tiles(data["tiles_coords"])

        input_batch = preprocess_tiles(tiles, width, height)
        out_batch = analyze_tiles(data["selected_model"], input_batch)

        if out_batch is not None:
            json = postprocess_tiles(
                out_batch=out_batch,
                width=width,
                height=height,
                start_tile_coords=data["tiles_coords"][0],
                analyze_area_polygon_dots=data["analyze_area_polygon"]
            )

            return jsonify(json)
        else:
            return jsonify({"polygons": None, "success": False, "message:": "Данная модель недоступна"})
    except Exception as ex:
        print(ex)

        return 'Not Found', 404
    """


def run_server(host: str, port: int, models: dict):
    global analyze_tiles
    global availible_models

    availible_models = models

    def _model_func(model_name: str, input_batch):
        if model_name in models:
            return availible_models[model_name](input_batch)
        else:
            return None

    analyze_tiles = _model_func

    print("Запуск Flask-приложения...")

    app.run(host=host, port=port)
