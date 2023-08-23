import io
import os.path
import time

import numpy as np
import tensorflow as tf
import datetime as dt

from livelossplot import PlotLosses
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from googleapiclient.discovery import build


WEIGHTS_DIR = "weights"
TMP_WEIGHTS_DIR = "tmp"
REMOTE_WEIGHTS_DIR_ID = "12yzyc54hrUGNYQLBVfejX38OdQBSnQLk"

SCOPES = ['https://www.googleapis.com/auth/drive']
CREDITS_FILE = "addition_data/google-api-credits.json"


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


def save_weights_local(model: tf.keras.models.Model) -> None:
    name = f"weights {model.name} {dt.datetime.now()}.h5"
    model.save_weights(os.path.join(WEIGHTS_DIR, name))


def save_weights_remote(model: tf.keras.models.Model) -> None:
    if not os.path.exists(TMP_WEIGHTS_DIR):
        os.mkdir(TMP_WEIGHTS_DIR)

    name = f"weights {model.name} {dt.datetime.now()}.h5"
    temp_weights_path = os.path.join(TMP_WEIGHTS_DIR, f"{time.time()}.h5")

    model.save_weights(temp_weights_path)

    credentials = service_account.Credentials.from_service_account_file(
        CREDITS_FILE,
        scopes=SCOPES
    )
    service = build('drive', 'v3', credentials=credentials)

    file_metadata = {
        'name': name,
        'parents': [REMOTE_WEIGHTS_DIR_ID]
    }

    media = MediaFileUpload(temp_weights_path, resumable=True)
    service.files().create(body=file_metadata, media_body=media, fields='if').execute()

    os.remove(temp_weights_path)


def get_local_weights_path(model_name: str) -> str | None:
    """
    Возвращает путь с самым последним локально сохраненным файлом весов для данной модели, None - если сохраненных весов
    нет
    """
    names = [i for i in os.listdir(WEIGHTS_DIR) if model_name in i]

    try:
        return os.path.join(WEIGHTS_DIR, names[np.argmax([_get_date_from_name(i) for i in names])])
    except:
        return None


def get_remote_weights_path(model_name: str):
    """
    Загружает последний сохраненный файл весов для данной модели и возвращает путь к загруженному файлу, None - если
    сохраненных весов нет или если при загрузке произошла ошибка
    """
    if not os.path.exists(TMP_WEIGHTS_DIR):
        os.mkdir(TMP_WEIGHTS_DIR)

    credentials = service_account.Credentials.from_service_account_file(
        CREDITS_FILE,
        scopes=SCOPES
    )

    service = build('drive', 'v3', credentials=credentials)

    file = service.files().list(
        pageSize=999,
        fields="files(id, name)",
        q=f"'{model_name}' in name",
        order_by="createdTime desc"
    ).execute()["files"]

    if len(file) == 0:
        return None
    else:
        try:
            weights_file_name = f"{time.time()}.h5"

            request = service.files().get_media(fileId=file[0]['id'])
            stream = io.FileIO(os.path.join(TMP_WEIGHTS_DIR, weights_file_name), 'wb')
            downloader = MediaIoBaseDownload(stream, request)

            done = False
            while done is False:
                status, done = downloader.next_chunk()

            return os.path.join(TMP_WEIGHTS_DIR, weights_file_name)
        except:
            return None