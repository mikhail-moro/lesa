import traceback

from flask import Flask, render_template, request, jsonify
from .tile_processing import preprocess_tiles, postprocess_tiles, load_tiles


app = Flask(__name__)


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()

    try:
        tiles_coords = data["tiles_coords"]
        start_tiles_coords = data["tiles_coords"][0]
        model_name = data["selected_model"]
        analyze_area_polygon = data["analyze_area_polygon"]

        tiles, width, height = load_tiles(tiles_coords)

        input_batch = preprocess_tiles(tiles, width, height)
        out_batch = tiles_analyzer[model_name](input_batch)

        if out_batch is not None:
            json = postprocess_tiles(
                out_batch=out_batch,
                width=width,
                height=height,
                start_tile_coords=start_tiles_coords,
                analyze_area_polygon_dots=analyze_area_polygon
            )

            return jsonify(json)
        else:
            return jsonify({"polygons": None, "success": False, "message:": "Данная модель недоступна"})
    except Exception:
        error_logger.log(log_type='error', request=str(data), traceback=traceback.format_exc())

        return 'Not Found', 404


def run_server(host: str, port: int, analyzer, logger):
    """Запуск Flask-приложения"""
    global tiles_analyzer
    global error_logger

    tiles_analyzer = analyzer
    error_logger = logger

    print("Запуск Flask-приложения...")

    app.run(host=host, port=port)
