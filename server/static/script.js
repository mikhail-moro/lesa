const TILES_URL = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}';
const META_URL = 'https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}'

// площади 1 пикселя тайла в квадратных метрах для каждого из доступных для анализа значения координаты z (zoom)
let tilePixAreas = {
    16: 2.6261664581298834,
    17: 0.6563178253173828,
    18: 0.1641913604736328
}

let mapOptions = {
    center: [47.2313, 39.7233],
    zoom: 12,
    minZoom: 12
};

let controlsOptions = {
    position: 'topleft',
    draw: {
        polygon: {
            allowIntersection: false,
            drawError: {
                color: '#e1e100',
                message: 'Неправильная форма'
            }
        },
        toolbar: {
            buttons: {
                polygon: 'Выделите область для анализа',
                rectangle: 'Выделите область для анализа'
            }
        },
        polygon: true,
        rectangle: true,
        circle: false,
        marker: false,
        polyline: false,
        circlemarker: false
    }
};

let bounds = L.latLngBounds(
    L.latLng(47.132631, 39.372211),
    L.latLng(47.375889, 39.873439)
);


L.Control.ModelSwitchControl = L.Control.extend({
    onAdd: function(map) {
        let control = L.DomUtil.create('div', 'control-model-switch');

        control.innerHTML = `
            <form>
                <div id="model-switch">
                    <label class="model-switch-block" id="radio-unet"><input type="radio" name="model-switch-group" value="unet">U-Net</label>
                    <label class="model-switch-block" id="radio-unet-plus-plus"><input type="radio" name="model-switch-group" value="unet_plus_plus">U-Net++</label>
                    <label class="model-switch-block" id="radio-deeplab-v3-plus"><input type="radio" name="model-switch-group" value="deeplab_v3_plus">DeepLabV3+</label>
                </div>
            </form>
        `;

        return control;
    },

    onRemove: function(map) {}
});

L.control.ModelSwitchControl = function(opts) {
    return new L.Control.ModelSwitchControl(opts);
};


let tilesLayer = new L.TileLayer(TILES_URL, mapOptions);
let metaLayer = new L.TileLayer(META_URL, mapOptions);
let featuresLayer = new L.FeatureGroup();


let map = new L.map('map', mapOptions);
map.setMaxBounds(bounds);
map.addLayer(tilesLayer);
map.addLayer(metaLayer);
map.addLayer(featuresLayer);
map.addControl(new L.Control.Draw(controlsOptions));
map.addControl(new L.control.ModelSwitchControl({position: 'topleft'}));


let leafletDrawToolbar = document.getElementsByClassName("leaflet-draw-toolbar leaflet-bar leaflet-draw-toolbar-top")[0];
let modelSwitch = document.getElementById("model-switch");

document.getElementById("radio-unet").children[0].checked = true;
leafletDrawToolbar.style.display = 'none';


map.on('drag', function() {
    map.panInsideBounds(bounds, { animate: false });
});

map.on('zoomstart', function() {
    featuresLayer.clearLayers()
});

map.on('draw:created', function(e) {
    featuresLayer.addLayer(e.layer);

    // узнаем крайние точки выделенной области
    let rect = e.layer._path.getBoundingClientRect();
    let minX = rect.x;
    let minY = rect.y;
    let maxX = rect.x + rect.width;
    let maxY = rect.y + rect.height;

    // узнаем тайлы крайних точек выделенной области
    let startTile = null;
    let endTile = null;

    for (let key in tilesLayer._tiles) {
        let cords = tilesLayer._tiles[key].el.getBoundingClientRect();

        if ((minX >= cords.x && minX < cords.x + cords.width) && (minY >= cords.y && minY < cords.y + cords.height)) {
            startTile = tilesLayer._tiles[key];
            break;
        }
    }
    for (let key in tilesLayer._tiles) {
        let cords = tilesLayer._tiles[key].el.getBoundingClientRect();

        if ((maxX >= cords.x && maxX < cords.x + cords.width) && (maxY >= cords.y && maxY < cords.y + cords.height)) {
            endTile = tilesLayer._tiles[key];
            break;
        }
    }

    // узнаем координаты всех тайлов исследуемой области
    var tiles_coords = [];

    let tileZ = startTile.coords['z'];
    let tileX = startTile.coords['x'];

    while (tileX <= endTile.coords['x']) {
        let tileY = startTile.coords['y'];

        while (tileY <= endTile.coords['y']) {
            tiles_coords.push({x: tileX, y: tileY, z: tileZ});
            tileY = tileY + 1;
        }

        tileX = tileX + 1;
    }

    let selectedModel = null;

    for (let label of modelSwitch.children) {
        let modelSwitchButton = label.children[0];

        if (modelSwitchButton.checked) {
            selectedModel = modelSwitchButton.value;
        }
    }

    // отправляем для анализа координаты задетых тайлов и точки полигона описывающего выделенную пользователем область
    fetch("http://127.0.0.1/analyze", {
        method: "POST",
        body: JSON.stringify({
            "tiles_coords": tiles_coords,
            "analyze_area_polygon": e.layer._latlngs[0],
            "selected_model": selectedModel
        }),
        headers: {
            "Content-type": "application/json; charset=UTF-8"
        }
    })
        .then(response => response.json())
        .then(data => {
            for (let polygonData of data["polygons"]) {
                let polygonArea = polygonData["area"] * tilePixAreas[tileZ];
                polygonArea = Number(polygonArea).toFixed(2)

                L.polygon(polygonData["coords"])
                    .addTo(map)
                    .bindPopup(polygonArea + " м²");
            }

            featuresLayer.removeLayer(e.layer);
        });
});

map.on('zoomend', function() {
    if (map.getZoom() > 15) {
        if (leafletDrawToolbar.style.display != null) {
            leafletDrawToolbar.style.display = null;
        }
    } else {
        if (leafletDrawToolbar.style.display != 'none') {
            leafletDrawToolbar.style.display = 'none';
        }
    }
});
