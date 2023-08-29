import asyncio
from io import BytesIO

import aiohttp
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt

#
# За быстрейшие параллельные запросы спасибо пользователю Pedro Serra
# https://stackoverflow.com/questions/57126286/fastest-parallel-requests-in-python
#

TILES_SERVER = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

TILE_SIZE = 256
PADDING_SIZE = 22


def get_tiles(tiles_cords: list) -> np.ndarray:
    """
    Загружает тайлы по координатам в виде массива следуюжего вида: np.ndarray[height, width, 256, 256, 3]

    :param tiles_cords: координаты тайлов в виде списка словарей вида: {'x': x, 'y': y, 'z': z}
    """
    xx = [i['x'] for i in tiles_cords]
    yy = [i['y'] for i in tiles_cords]

    width = max(xx) - min(xx) + 1
    height = max(yy) - min(yy) + 1

    _tiles = {}

    async def _http_worker(cords, session, index):
        # try
        async with session.get(url=TILES_SERVER.format(**cords)) as response:
            _tiles[index] = np.asarray(Image.open(BytesIO(await response.read())), dtype=np.uint8)

    async def main():
        async with aiohttp.ClientSession() as session:
            await asyncio.gather(
                *[_http_worker(cords, session, i) for i, cords in enumerate(tiles_cords)]
            )

    asyncio.run(main())

    _tiles = sorted(_tiles.items())

    return np.array([i[1] for i in _tiles]).reshape((height, width, TILE_SIZE, TILE_SIZE, 3))


def preprocess_tiles(tiles_cords: list) -> np.ndarray:
    xx = [i['x'] for i in tiles_cords]
    yy = [i['y'] for i in tiles_cords]

    width = max(xx) - min(xx) + 1
    height = max(yy) - min(yy) + 1

    # загрузка тайлов с сервера

    tiles_queue = {}

    async def _http_worker(cords, session, index):
        # try
        async with session.get(url=TILES_SERVER.format(**cords)) as response:
            tiles_queue[index] = np.asarray(Image.open(BytesIO(await response.read())), dtype=np.uint8)

    async def main():
        async with aiohttp.ClientSession() as session:
            await asyncio.gather(
                *[_http_worker(cords, session, i) for i, cords in enumerate(tiles_cords)]
            )

    asyncio.run(main())

    tiles_queue = sorted(tiles_queue.items())
    tiles_data = np.array([i[1] for i in tiles_queue]).reshape((height, width, TILE_SIZE, TILE_SIZE, 3))

    # для того, чтобы модель лучше обрабатывала пиксели по краям тайлов, необходимо нарезать их с захватом контекста с
    # соседних тайлов

    empty_map = np.zeros(((height*TILE_SIZE)+PADDING_SIZE*2, (width*TILE_SIZE)+PADDING_SIZE*2, 3))
    stacked_tiles = np.vstack([np.hstack([tiles_data[j][i] for i in np.arange(width)]) for j in np.arange(height)])

    empty_map[PADDING_SIZE:stacked_tiles.shape[0] + PADDING_SIZE, PADDING_SIZE:stacked_tiles.shape[1] + PADDING_SIZE] = stacked_tiles

    tiles = []

    for y in np.arange(PADDING_SIZE, height*TILE_SIZE+PADDING_SIZE, TILE_SIZE):
        for x in np.arange(PADDING_SIZE, width*TILE_SIZE+PADDING_SIZE, TILE_SIZE):
            x_0 = x - PADDING_SIZE
            y_0 = y - PADDING_SIZE
            x_1 = x + PADDING_SIZE + TILE_SIZE
            y_1 = y + PADDING_SIZE + TILE_SIZE

            tiles.append(empty_map[y_0:y_1, x_0:x_1])

    return np.array(tiles).reshape((1, len(tiles), TILE_SIZE+PADDING_SIZE*2, TILE_SIZE+PADDING_SIZE*2, 3)) / 255


def postprocess(out_batch: list):
    result = []

    for mask in out_batch:
        im_arr = []

        for x in mask:
            xx = []

            for y in x:
                xx.append([0, 0, 0, y*255])

            im_arr.append(xx)

        result.append(Image.fromarray(np.asarray(im_arr, dtype=np.uint8), 'RGBA'))

    return result


if __name__ == "__main__":
    data = preprocess_tiles([
        {'x': 39970, 'y': 23004, 'z': 16},
        {'x': 39971, 'y': 23004, 'z': 16},
        {'x': 39970, 'y': 23005, 'z': 16},
        {'x': 39971, 'y': 23005, 'z': 16},
        {'x': 39970, 'y': 23006, 'z': 16},
        {'x': 39971, 'y': 23006, 'z': 16}
    ])[0]

    for t in range(6):
        plt.subplot(321+t).imshow(data[t])

    plt.show()
