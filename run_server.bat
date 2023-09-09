@SET HOST=127.0.0.1
@SET /a PORT=80


:: Расположение весов для моделей: ['local', 'remote']
::   'local' - брать веса для моделей из локальной диреектории
::   'remote' - брать веса для моделей из удаленной директории
@SET WEIGHTS_LOCATION=remote

:: Локальная директория с весами
@SET LOCAL_WEIGHTS_DIR_PATH=weights

:: ID удаленной Google Drive директории с весами
@SET REMOTE_WEIGHTS_GOOGLE_DRIVE_DIR_ID=12yzyc54hrUGNYQLBVfejX38OdQBSnQLk

:: Список используемых моделей в формате: -m model_name_1 -m model_name_2 ...
@SET MODELS=-m unet -m unet_plus_plus -m deeplab_v3_plus


:: Путь к Python-интерпритатору
@SET PY=python

%PY% main.py %HOST% %PORT% %WEIGHTS_LOCATION% %LOCAL_WEIGHTS_DIR_PATH% %REMOTE_WEIGHTS_GOOGLE_DRIVE_DIR_ID% %MODELS%

pause