const TILES_URL = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}';
const META_URL = 'https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}'
const MASKS_URL = "http://127.0.0.1/tile?z={z}&y={y}&x={x}";

var mapOptions = {
    center: [47.2313, 39.7233],
    zoom: 12,
    minZoom: 12
};

var map = new L.map('map', mapOptions);

var featuresLayer = new L.FeatureGroup();
map.addLayer(featuresLayer);

var controlsOptions = {
    position: 'topleft',
    draw: {
        polygon: {
            allowIntersection: false,
            drawError: {
                color: '#e1e100',
                message: 'Неправильная форма'
            },
            shapeOptions: {
                color: '#97009c'
            }
        },
        circle: true,
        polygon: true,
        rectangle: true,
        marker: false,
        polyline: false,
        circlemarker: false
    },
    edit: {
        featureGroup: featuresLayer,
        remove: true
    }
};


var drawControl = new L.Control.Draw(controlsOptions);
var southWest = L.latLng(47.132631, 39.372211);
var northEast = L.latLng(47.375889, 39.873439);
var bounds = L.latLngBounds(southWest, northEast);
var tilesLayer = new L.TileLayer(TILES_URL, mapOptions);
var metaLayer = new L.TileLayer(META_URL, mapOptions);
var masksLayer = new L.TileLayer(MASKS_URL, mapOptions);


map.setMaxBounds(bounds);
map.addLayer(tilesLayer);
map.addLayer(metaLayer);
map.addLayer(masksLayer);

map.on('drag', function() {
    map.panInsideBounds(bounds, { animate: false });
});

map.on('draw:created', function(e) {
    featuresLayer.addLayer(e.layer);


    // высчитываем крайние точки полигона
    var maxX = -9999;
    var maxY = -9999;
    var minX = +9999;
    var minY = +9999;

    for (let coords of e.layer._parts[0]) {
        if (coords['x'] > maxX) {
            maxX = coords["x"];
        }
        if (coords['y'] > maxY) {
            maxY = coords["y"];
        }
        if (coords['x'] < minX) {
            minX = coords["x"];
        }
        if (coords['y'] < minY) {
            minY = coords["y"];
        }
    }


    // узнаем тайлы крайних точек полигона
    var maxTile = null;
    var minTile = null;

    for (let key in tilesLayer._tiles) {
        let tile = tilesLayer._tiles[key];
        let poses = tile.el._leaflet_pos;

        if ((maxX >= poses['x'] && maxX < poses['x'] + 256) && (maxY >= poses["y"] && maxY < poses['y'] + 256)) {
            maxTile = tile;
            break;
        }
    }
    for (let key in tilesLayer._tiles) {
        let tile = tilesLayer._tiles[key];
        let poses = tile.el._leaflet_pos;

        if ((minX >= poses['x'] && minX < poses['x'] + 256) && (minY >= poses["y"] && minY < poses['y'] + 256)) {
            minTile = tile;
            break;
        }
    }


    // узнаем точки полигона относительно исследуемой области
    let startX = minTile.el._leaflet_pos['x'];
    let startY = minTile.el._leaflet_pos['y'];

    var polygon_dots = [];

    for (let dot of e.layer._parts[0]) {
        polygon_dots.push({x: dot['x'] - startX, y: dot["y"] - startY});
    }


    // узнаем координаты всех тайлов исследуемой области
    var tiles_coords = [];

    var tileZ = minTile.coords['z'];
    var tileX = minTile.coords['x'];

    while (tileX <= maxTile.coords['x']) {
        var tileY = minTile.coords['y'];

        while (tileY <= maxTile.coords['y']) {
            tiles_coords.push({x: tileX, y: tileY, z: tileZ});

            tileY = tileY + 1;
        }

        tileX = tileX + 1;
    }

    fetch("http://127.0.0.1/analyze", {
        method: "POST",
        body: JSON.stringify({
            "tiles_coords": tiles_coords,
            "polygon_dots": polygon_dots
        }),
        headers: {
            "Content-type": "application/json; charset=UTF-8"
        }
    })
        .then((_) => masksLayer.redraw());
});

map.on("click", (e) => {
    console.log(e.layerPoint);
})

map.addControl(drawControl);

var command = L.control({position: 'topright'});

command.onAdd = function (_) {
    var div = L.DomUtil.create('div', 'command');

    div.innerHTML = '<form><input id="command" type="checkbox"/>command</form>';
    return div;
};

command.addTo(map);
