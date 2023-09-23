import io
import os.path
import time

import numpy as np
import datetime as dt

from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.discovery import build

from . import TMP_WEIGHTS_DIR_PATH


SCOPES = ['https://www.googleapis.com/auth/drive']


def _get_now_date():
    return dt.datetime.now().strftime("%y-%m-%d %H-%M")


def _get_date_from_name(name):
    return dt.datetime.strptime(name[-14], "%y-%m-%d %H-%M")


def get_local_weights_path(model_name: str, weights_dir_path: str) -> str | None:
    """
    Возвращает путь с самым последним локально сохраненным файлом весов для данной модели, None - если сохраненных весов
    нет
    """
    names = [i for i in os.listdir(weights_dir_path) if model_name in i]

    try:
        return os.path.join(weights_dir_path, names[np.argmax([_get_date_from_name(i) for i in names])])
    except Exception as ex:
        print(ex)

        raise ValueError("Server Error: Ошибка загрузки весов")


def get_remote_weights_path(model_name: str, weights_dir_id: str, credentials_file_path: str):
    """
    Загружает последний сохраненный файл весов для данной модели и возвращает путь к загруженному файлу, None - если
    сохраненных весов нет или если при загрузке произошла ошибка
    """
    if not os.path.exists(TMP_WEIGHTS_DIR_PATH):
        os.mkdir(TMP_WEIGHTS_DIR_PATH)

    credentials = service_account.Credentials.from_service_account_file(
        credentials_file_path,
        scopes=SCOPES
    )

    service = build('drive', 'v3', credentials=credentials)

    file = service.files().list(
        pageSize=999,
        fields="files(id, name)",
        q=f"'{weights_dir_id}' in parents and name contains '{model_name}'",
        orderBy="createdTime desc"
    ).execute()["files"]

    if len(file) == 0:
        return None
    else:
        try:
            weights_file_name = f"{int(time.time())}.h5"

            request = service.files().get_media(fileId=file[0]['id'])
            stream = io.FileIO(os.path.join(TMP_WEIGHTS_DIR_PATH, weights_file_name), 'wb')
            downloader = MediaIoBaseDownload(stream, request)

            done = False
            while done is False:
                status, done = downloader.next_chunk()

            return os.path.join(TMP_WEIGHTS_DIR_PATH, weights_file_name)
        except Exception as ex:
            print(ex)

            raise ValueError("Server Error: Ошибка загрузки весов")


def remove_models_temps():
    for file in os.listdir(TMP_WEIGHTS_DIR_PATH):
        os.remove(os.path.join(TMP_WEIGHTS_DIR_PATH, file))
