# Маркировка и оценка изменений лесопарковых площадей

Данный сервис позволит анализировать изменения лесопарковых площадей Ростова-на-Дону


### Дорожная карта:

  - :white_check_mark: Минимально работающий Web-клиент для интерактивного анализа

  - :arrow_right: Реализация основных архитектур FCCNN - Unet и DeepLab:
      - :white_check_mark: Обучены пробные версии моделей, пока не показывающие достаточной точности
      - :white_check_mark: Реализована интеграция моделей с Web-клиентом
      - ...
  
  - :x: Выбор баз снимков дистанционного зондирования Земли
  
  - :x: Выбор периода
  
  - :x: Отчет с аналитикой

### Реализованные архитектуры
- U-Net<sup>[[1]](https://arxiv.org/abs/1505.04597)</sup> - доступно в Web-клиенте
- UNet++<sup>[[2]](https://arxiv.org/abs/1807.10165)</sup> - доступно в Web-клиенте
- DeepLabV3+<sup>[[3]](https://arxiv.org/abs/1802.02611)</sup>:
  - версия с EfficientNetB3 енкодером - доступно в Web-клиенте
  - версия с ResNet50 енкодером - дорабатывается, пока не доступно в Web-клиенте

Все модели обучались на следующем датасете: https://www.kaggle.com/datasets/quadeer15sh/augmented-forest-segmentation
 
На рис. 1 показано сравнение метрики Intersection-Over-Union вывода моделей и размеченных вручную масок (данные для сравнения расположенны в models
/benchmark_data). Как видно, на данный момент, самые лучшие результаты показала архитектура DeepLabv3+** на уровне приближения 17 (модели анализируют изображения только с последних 3-х самых близких уровней приближения - 16, 17, 18). Однако, размеченных данных крайне мало, поэтому результаты нельзя назвать точными.

<p align="center"><img src="https://github.com/mikhail-moro/res/blob/main/models_comprassion.png"><br>Рис. 1</p>

На рис. 2 и рис. 3 видно, что модель (DeepLabv3+**) не идеально выделяет лесопарковые площади, особенно в городской среде (однако, тут играет роль и процесс постпроцессинга вывода модели, это тоже не доработано).

<p align="center"><img src="https://github.com/mikhail-moro/res/blob/main/sample_1.png"><br>Рис. 2</p>
<p align="center"><img src="https://github.com/mikhail-moro/res/blob/main/sample_2.png"><br>Рис. 3</p>

-----
** Использовалась версия DeepLabV3+ с ResNet50 енкодером
 
1. https://arxiv.org/abs/1505.04597
2. https://arxiv.org/abs/1807.10165
3. https://arxiv.org/abs/1802.02611

### Демонстрация работы сервиса (ссылка на YouTube-видео)
[![Демонстрация работы](https://img.youtube.com/vi/okUjgAhp0fM/maxresdefault.jpg)](https://www.youtube.com/watch?v=okUjgAhp0fM)

### Запуск сервиса
Windows:
  - запустить ```install_requirements.bat``` для загрузки зависимостей через pip
  - запустить ```run_server.bat``` для запуска сервера

Параметры запуска расписаны в .bat файлах

*пока только Windows*

### TODO
  - доработать unit-тесты
  - исправить баг web-клиента, когда выделенная область не анализируется если находится, хотя бы частично, вне экрана
  - исправить ещё кучу багов клиента
  - добавить размеченных примеров для сравнения моделей
  - ...
