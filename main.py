import argparse
import atexit
import os.path

from server import remove_server_temps, run_server
from models import remove_models_temps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('server_host', type=str)
    parser.add_argument('server_port', type=int)
    parser.add_argument('weights_location', type=str)
    parser.add_argument('local_weights_dir_path', type=str)
    parser.add_argument('remote_weights_dir_id', type=str)
    parser.add_argument('-m', '--models', type=list[str], action='append')

    args = parser.parse_args()

    # TODO добавить проверку аргументов

    if args.weights_location == 'local':
        model_kwargs = {
            "weights_file": 'auto',
            "weights_dir": os.path.join(__file__[-8], args.local_weights_dir_path),
            "weights_destination": args.weights_location
        }
    elif args.weights_location == 'remote':
        model_kwargs = {
            "weights_file": 'auto',
            "weights_dir": os.path.join(__file__[-8], args.remote_weights_dir_id),
            "weights_destination": args.weights_location
        }
    else:
        pass

    availible_models = {}

    for model in [''.join(i) for i in args.models]:  # argparse возвращает названия моделей в виде списков char объектов
        print(f"Инициализация {model}...")

        if model == 'unet':
            from models import UnetAnalyzer

            availible_models['unet'] = UnetAnalyzer(**model_kwargs)
        if model == 'unet_plus_plus':
            from models import UnetPlusPlusAnalyzer

            availible_models['unet_plus_plus'] = UnetPlusPlusAnalyzer(**model_kwargs)
        if model == 'deeplab_v3_plus':
            from models import DeeplabV3plusAnalyzer

            availible_models['deeplab_v3_plus'] = DeeplabV3plusAnalyzer(**model_kwargs)

    atexit.register(lambda: [func.__call__() for func in [remove_server_temps, remove_models_temps]])

    run_server(args.server_host, args.server_port, availible_models)
