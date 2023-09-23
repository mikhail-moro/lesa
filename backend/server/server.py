import typing
import traceback

if typing.TYPE_CHECKING:
    from lesa.backend.models import Analyzer

from flask import Flask, render_template, request, jsonify
from .tile_processing import preprocess_tiles, postprocess_tiles, uniq_coords, check_coords, TilesDownloader
from .utils import Logger
from . import STATIC_DIR_PATH, TEMPLATE_DIR_PATH


class Server(Flask):
    """
    Класс-обертка Flask-приложения

    :param analyzer: класс реализующий функционал анализа входящих тайлов
    :param logs_file_path: путь к .txt файлу где будут писаться логи сервера
    :param tiles_download_max_replies: во время асинхронной загрузки тайлов с сервера они могут загрузиться не полностью, параметр устанавливает сколько раз повторять запрос
    """
    def __init__(
        self,
        *flask_args,
        analyzer: 'Analyzer',
        logs_file_path: str,
        **flask_kwargs
    ):
        print("Запуск сервера...")

        super().__init__(*flask_args, **flask_kwargs, static_folder=STATIC_DIR_PATH, template_folder=TEMPLATE_DIR_PATH)

        logger = Logger(logs_file_path)
        downloader = TilesDownloader()

        @self.route('/')
        def main():
            return render_template('index.html', models=analyzer.get_models_names())

        @self.route('/analyze', methods=['POST'])
        def analyze():
            data = request.get_json()

            tiles_coords = data["tiles_coords"]
            start_tile_coords = data["tiles_coords"][0]
            model_name = data["selected_model"]
            analyze_area_polygon = data["analyze_area_polygon"]

            try:
                tiles_coords = uniq_coords(tiles_coords)
                width, height = check_coords(tiles_coords)

                tiles = downloader.get_tiles(tiles_coords)

                input_batch = preprocess_tiles(tiles, width, height)
                out_batch = analyzer[model_name](input_batch)

                polygons = postprocess_tiles(
                    out_batch=out_batch,
                    width=width,
                    height=height,
                    start_tile_coords=start_tile_coords,
                    analyze_area_polygon_dots=analyze_area_polygon
                )

                return jsonify({"polygons": polygons, "success": True, "message": None})

            except ValueError as ve:
                if str(ve).startswith("Client Error"):
                    return jsonify({"polygons": None, "success": False, "message": str(ve)})
                else:
                    print(ve)
                    logger.log(log_type='Exception', request=str(data), traceback=traceback.format_exc())

                    return 'Not Found', 404

            except Warning as wn:
                print(wn)
                logger.log(log_type='Warning', request=str(data), traceback=traceback.format_exc())

            except Exception as ex:
                print(ex)
                logger.log(log_type='Exception', request=str(data), traceback=traceback.format_exc())

                return 'Not Found', 404
