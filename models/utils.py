import io
import os.path
import time
import sys

import numpy as np
import tensorflow as tf
import datetime as dt

from livelossplot import PlotLosses
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from googleapiclient.discovery import build


# Данный скрипт может быть запущен как из файлов main.py и tests.py (стандартным способом), так и отдельно импортирован,
# например для обучения в Google Colab, в таком случае использовать относительные импорты не получиться
SCRIPT_RUN_SEPARATE = sys.path[0] == __file__[:-9]

if not SCRIPT_RUN_SEPARATE:
    from . import WEIGHTS_DIR_PATH, TMP_WEIGHTS_DIR_PATH


SCOPES = ['https://www.googleapis.com/auth/drive']


class PlotLossesCallback(tf.keras.callbacks.Callback):
    def __init__(self, clip: tuple = None, **kwargs):
        super().__init__()
        self.liveplot = PlotLosses(**kwargs)
        self.clip = clip

    def on_epoch_end(self, epoch, logs=None):
        limited_logs = logs.copy()

        if self.clip:
            for key, value in limited_logs.items():
                if value < self.clip[0]:
                    limited_logs[key] = self.clip[0]

                if value > self.clip[1]:
                    limited_logs[key] = self.clip[1]

        self.liveplot.update(limited_logs, epoch)
        self.liveplot.send()


def _get_now_date():
    return dt.datetime.now().strftime("%y-%m-%d %H-%M")


def _get_date_from_name(name):
    return dt.datetime.strptime(name[-14], "%y-%m-%d %H-%M")


def save_weights_local(model: tf.keras.models.Model, local_dir_path: str) -> None:
    name = f"weights {model.name} {dt.datetime.now()}.h5"
    model.save_weights(os.path.join(local_dir_path, name))


def save_weights_remote(model: tf.keras.models.Model, remote_dir_id: str, credentials_file_path: str) -> None:
    if not os.path.exists(TMP_WEIGHTS_DIR_PATH):
        os.mkdir(TMP_WEIGHTS_DIR_PATH)

    name = f"weights {model.name} {dt.datetime.now()}.h5"
    temp_weights_path = os.path.join(TMP_WEIGHTS_DIR_PATH, f"{int(time.time())}.h5")

    model.save_weights(temp_weights_path)

    credentials = service_account.Credentials.from_service_account_file(
        credentials_file_path,
        scopes=SCOPES
    )
    service = build('drive', 'v3', credentials=credentials)

    file_metadata = {
        'name': name,
        'parents': [remote_dir_id]
    }

    media = MediaFileUpload(temp_weights_path, resumable=True)
    service.files().create(body=file_metadata, media_body=media, fields='id').execute()


def get_local_weights_path(model_name: str, weights_dir_path: str) -> str | None:
    """
    Возвращает путь с самым последним локально сохраненным файлом весов для данной модели, None - если сохраненных весов
    нет
    """
    names = [i for i in os.listdir(WEIGHTS_DIR_PATH) if model_name in i]

    try:
        return os.path.join(weights_dir_path, names[np.argmax([_get_date_from_name(i) for i in names])])
    except Exception as ex:
        print(ex)

        return None


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

            return None


def remove_models_temps():
    for file in os.listdir(TMP_WEIGHTS_DIR_PATH):
        os.remove(os.path.join(TMP_WEIGHTS_DIR_PATH, file))
