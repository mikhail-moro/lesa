import asyncio
from io import BytesIO

import aiohttp
import numpy as np

from PIL import Image

#
# За быстрейшие параллельные запросы спасибо пользователю Pedro Serra
# https://stackoverflow.com/questions/57126286/fastest-parallel-requests-in-python
#

TILES_SERVER = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
PADDING_SIZE = 22


def get_tiles(tiles_cords: list[dict[str:int]]):
    _tiles = {}

    async def get(url, session, index):
        # try
        async with session.get(url=url) as response:
            _tiles[index] = Image.open(BytesIO(await response.read()))

    async def main():
        async with aiohttp.ClientSession() as session:
            await asyncio.gather(
                *[get(TILES_SERVER.format(**cords), session, i) for i, cords in enumerate(tiles_cords)]
            )

    asyncio.run(main())

    return np.asarray([np.asarray(_tiles[i]) for i in sorted(_tiles)]).reshape((len(_tiles), 256, 256, 3))


def model_out_to_png(out_batch: list):
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
