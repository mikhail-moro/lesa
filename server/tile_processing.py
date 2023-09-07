import array
import asyncio
import aiohttp

import numpy as np
import shapely
import tensorflow as tf

from shapely import Polygon
from PIL import Image
from io import BytesIO
from skimage import restoration, measure


TILES_SERVER = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

TILE_SIZE = 256  # Ширина и высота тайла
PADDING_SIZE = 16  # Размер отступа при подаче для анализа
BIN_TRESHOLD = .5  # Порог используемый при бинаризации выхода модели (матрица значений от 0 до 1)

MIN_POLYGON_AREA = (TILE_SIZE * 0.05) * (TILE_SIZE * 0.05)  # Минимальная площадь полигона (5% тайла - by default)
DECREASED_PADDING_SIZE = int(PADDING_SIZE * (TILE_SIZE / (TILE_SIZE + PADDING_SIZE * 2)))


resize = tf.keras.layers.Resizing(TILE_SIZE, TILE_SIZE)
rescale = tf.keras.layers.Rescaling(1. / 255.)
crop = tf.keras.layers.CenterCrop(TILE_SIZE - DECREASED_PADDING_SIZE * 2, TILE_SIZE - DECREASED_PADDING_SIZE * 2)


# https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
def _latlngs_to_pix_coords(lat: list, lng: list, start_tile_coords: dict):
    lat = np.array(lat)
    lng = np.array(lng)

    lat_rad = np.radians(lat)
    _2z = 2 ** start_tile_coords['z']

    tile_x = (lng + 180.0) / 360.0 * _2z
    tile_y = (1.0 - (np.log(np.tan(lat_rad) + (1/np.cos(lat_rad)))) / np.pi) * (_2z / 2)  # 2^(z-1)

    x = (tile_x - start_tile_coords['x']) * TILE_SIZE  # координаты тайлов в относительные пиксельные координаты
    y = (tile_y - start_tile_coords['y']) * TILE_SIZE

    return np.dstack((x, y))[0]


# https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
def _pix_coords_to_latlngs(x: array.array, y: array.array, start_tile_coords: dict) -> np.ndarray:
    x = np.array(x)
    y = np.array(y)

    x = x / TILE_SIZE + start_tile_coords['x']
    y = y / TILE_SIZE + start_tile_coords['y']
    _2z = 2 ** start_tile_coords['z']

    n = np.pi - 2 * np.pi * y / _2z

    lat = 180 / np.pi * np.arctan(0.5 * (np.exp(n) - np.exp(-n)))
    lng = x / _2z * 360 - 180

    return np.dstack((lat, lng))


def load_tiles(tiles_cords: list) -> (np.ndarray, int, int):
    xx = [i['x'] for i in tiles_cords]
    yy = [i['y'] for i in tiles_cords]

    width = max(xx) - min(xx) + 1
    height = max(yy) - min(yy) + 1

    tiles_queue = {}

    async def _http_worker(cords, session):
        # try
        async with session.get(url=TILES_SERVER.format(**cords)) as response:
            tiles_queue[tuple(cords.values())] = np.asarray(Image.open(BytesIO(await response.read())), dtype=np.uint8)

    async def _main():
        async with aiohttp.ClientSession() as session:
            await asyncio.gather(
                *[_http_worker(cords, session) for cords in tiles_cords]
            )

    asyncio.run(_main())

    # https://stackoverflow.com/questions/37111798/how-to-sort-a-list-of-x-y-coordinates
    tiles_queue = sorted(tiles_queue.items(), key=lambda i: (i[0][1], i[0][0]))

    return np.array([i[1] for i in tiles_queue]), width, height


def preprocess_tiles(tiles_data: np.ndarray, width, height) -> tf.Tensor:
    # для того, чтобы модель лучше обрабатывала пиксели по краям тайлов, необходимо нарезать их с захватом контекста с
    # соседних тайлов
    empty_frame = np.zeros(((height * TILE_SIZE) + PADDING_SIZE * 2, (width * TILE_SIZE) + PADDING_SIZE * 2, 3))
    stacked_tiles = np.vstack([np.hstack([tiles_data[i*width+j] for j in np.arange(width)]) for i in np.arange(height)])

    frame_height = stacked_tiles.shape[0]
    frame_width = stacked_tiles.shape[1]

    empty_frame[PADDING_SIZE:frame_height + PADDING_SIZE, PADDING_SIZE:frame_width + PADDING_SIZE] = stacked_tiles

    tiles = []

    for y in np.arange(0, height * TILE_SIZE, TILE_SIZE):
        for x in np.arange(0, width * TILE_SIZE, TILE_SIZE):
            tiles.append(empty_frame[y:y + PADDING_SIZE * 2 + TILE_SIZE, x:x + PADDING_SIZE * 2 + TILE_SIZE])

    return tf.convert_to_tensor([resize(rescale(i)) for i in tiles])


def postprocess_tiles(
    out_batch: tf.Tensor,
    width: int,
    height: int,
    start_tile_coords: dict,
    analyze_area_polygon_dots: list[dict]
) -> dict:
    stacked_masks = np.vstack([np.hstack([out_batch[i*width+j] for j in np.arange(width)]) for i in np.arange(height)])

    denoised_masks = restoration.denoise_nl_means(
        stacked_masks.reshape((stacked_masks.shape[0], stacked_masks.shape[1])),
        patch_size=8,
        patch_distance=16
    )
    denoised_masks[denoised_masks < BIN_TRESHOLD] = 0.
    denoised_masks[denoised_masks >= BIN_TRESHOLD] = 1.
    contours = measure.find_contours(denoised_masks)

    analyze_area_polygon_dots = [tuple([i[1] for i in dot.items()]) for dot in analyze_area_polygon_dots]
    analyze_area_polygon_dots = _latlngs_to_pix_coords(
        [i[0] for i in analyze_area_polygon_dots],
        [i[1] for i in analyze_area_polygon_dots],
        start_tile_coords=start_tile_coords
    )
    analyze_area_polygon = Polygon(analyze_area_polygon_dots)  # полигон описывающий выделенную пользователем область

    polygons = []

    for contour in contours:
        # из-за различий в системах координат используемых при вичислении контуров и в leaflet картах необходимо
        # поменять x y местами
        contour[:, [0, 1]] = contour[:, [1, 0]]

        polygon = Polygon(contour)
        polygon = polygon.simplify(tolerance=0.5)

        if polygon.area > MIN_POLYGON_AREA:
            # чтобы полигон не выходил за выделенную пользователем область (описываемую полигоном analyze_area_polygon)
            # возьмём их пересечение
            polygon = analyze_area_polygon & polygon

            if isinstance(polygon, shapely.Polygon):
                polygon_area = polygon.area

                polygon = _pix_coords_to_latlngs(
                    *polygon.exterior.xy,
                    start_tile_coords=start_tile_coords
                )

                polygons.append({
                    "coords": polygon.astype(float).tolist(),  # numpy объекты не serializable
                    "area": float(polygon_area)
                })
            else:
                # в некоторых случаях пересечение полигонов создает класс shapely.MultiPolygon, из которого нужно в
                # цикле извлекать еденичные полигоны
                for geom in polygon.geoms:
                    geom_area = polygon.area

                    geom = _pix_coords_to_latlngs(
                        *geom.exterior.xy,
                        start_tile_coords=start_tile_coords
                    )

                    polygons.append(geom.astype(float).tolist())
                    polygons.append({
                        "coords": geom.astype(float).tolist(),
                        "area": float(geom_area)
                    })

    return {"polygons": polygons}
