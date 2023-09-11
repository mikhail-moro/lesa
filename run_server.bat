@SET HOST=127.0.0.1
@SET /a PORT=80


:: Не используйте абсолютные пути, все пути должны быть относительны данной директории


:: Расположение файла с логами ошибок
@SET LOGS_FILE_PATH=logs.txt

:: Расположение весов для моделей: ['local', 'remote']
::     'local' - использовать веса для моделей из локальной директории
::     'remote' - использовать веса для моделей из удаленной директории
@SET WEIGHTS_LOCATION=remote

:: Локальная директория с файлами весов
@SET LOCAL_WEIGHTS_DIR_PATH=local_weights

:: ID удаленной Google Drive директории с файлами весов
@SET REMOTE_WEIGHTS_GOOGLE_DRIVE_DIR_ID=12yzyc54hrUGNYQLBVfejX38OdQBSnQLk
:: Путь к json-файлу с API-KEY для доступа к выше описанной Google Drive директории
@SET REMOTE_WEIGHTS_GOOGLE_DRIVE_CREDENTIALS=addition_data/google-api-credits.json

:: Список используемых моделей в формате: -m model_name_1 -m model_name_2 ...
@SET MODELS=-m unet -m unet_plus_plus -m deeplab_v3_plus


:: Путь к Python-интерпритатору
@SET PY=python

%PY% main.py %HOST% %PORT% %LOGS_FILE_PATH% %WEIGHTS_LOCATION% %LOCAL_WEIGHTS_DIR_PATH% %REMOTE_WEIGHTS_GOOGLE_DRIVE_DIR_ID% %REMOTE_WEIGHTS_GOOGLE_DRIVE_CREDENTIALS% %MODELS%

pause