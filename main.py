import argparse
import atexit
import os.path

from server import remove_server_temps, Server
from models import remove_models_temps, Analyzer


MAIN_PATH = __file__[:-8]  # убираем \main.py


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('server_host', type=str)
    parser.add_argument('server_port', type=int)
    parser.add_argument('requests_max_retries', type=int)
    parser.add_argument('logs_file_path', type=str)
    parser.add_argument('weights_location', type=str)
    parser.add_argument('local_weights_dir_path', type=str)
    parser.add_argument('remote_weights_dir_id', type=str)
    parser.add_argument('remote_weights_credentials_path', type=str)
    parser.add_argument('-m', '--models', type=list[str], action='append')

    args = parser.parse_args()

    # TODO добавить проверку аргументов

    # argparse возвращает названия моделей в виде списков char объектов
    selected_models = [''.join(i) for i in args.models]

    if args.weights_location == 'local':
        analyzer_kwargs = {
            "weights_file": 'auto',
            "weights_dir": os.path.join(MAIN_PATH, args.local_weights_dir_path),
            "weights_destination": args.weights_location,
            "google_drive_credentials_path": None
        }
    elif args.weights_location == 'remote':
        analyzer_kwargs = {
            "weights_file": 'auto',
            "weights_dir": args.remote_weights_dir_id,
            "weights_destination": args.weights_location,
            "google_drive_credentials_path": args.remote_weights_credentials_path
        }
    else:
        analyzer_kwargs = {}

    analyzer = Analyzer(selected_models, **analyzer_kwargs)

    app = Server(
        import_name=__name__,
        analyzer=analyzer,
        logs_file_path=os.path.join(MAIN_PATH, args.logs_file_path),
        tiles_download_max_replies=5
    )

    # удаление временных файлов при завершении скрипта
    atexit.register(lambda: [func() for func in [remove_server_temps, remove_models_temps]])

    app.run(args.server_host, args.server_port)
