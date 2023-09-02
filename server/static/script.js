const TILES_URL = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}';
const META_URL = 'https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}'
const MASKS_URL = "http://127.0.0.1/tile?z={z}&y={y}&x={x}";


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
            },
            shapeOptions: {
                color: '#97009c'
            }
        },
        polygon: false,
        rectangle: true,
        circle: false,
        marker: false,
        polyline: false,
        circlemarker: false
    }
};


let bounds = L.latLngBounds(L.latLng(47.132631, 39.372211), L.latLng(47.375889, 39.873439));
let tilesLayer = new L.TileLayer(TILES_URL, mapOptions);
let metaLayer = new L.TileLayer(META_URL, mapOptions);
let featuresLayer = new L.FeatureGroup();
let masksLayer = new L.TileLayer(MASKS_URL, mapOptions);


let map = new L.map('map', mapOptions);
map.setMaxBounds(bounds);
map.addLayer(tilesLayer);
map.addLayer(metaLayer);
map.addLayer(featuresLayer);
map.addLayer(masksLayer);

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
    var startTile = null;
    var endTile = null;

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


    /*
    // узнаем точки полигона относительно исследуемой области
    let startX = startTile.el._leaflet_pos['x'];
    let startY = startTile.el._leaflet_pos['y'];

    var polygon_dots = [];

    for (let dot of e.layer._parts[0]) {
        polygon_dots.push({x: dot['x'] - startX, y: dot["y"] - startY});
    }
    */

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


    fetch("http://127.0.0.1/analyze", {
        method: "POST",
        body: JSON.stringify({
            "tiles_coords": tiles_coords,
            //"polygon_dots": polygon_dots
        }),
        headers: {
            "Content-type": "application/json; charset=UTF-8"
        }
    })
        .then(() => masksLayer.redraw());
});


// let leafletDrawToolbar = document.getElementsByClassName("leaflet-draw-toolbar leaflet-bar leaflet-draw-toolbar-top")[0];
let drawControl = new L.Control.Draw(controlsOptions);
map.addControl(drawControl);

/*
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

leafletDrawToolbar.style.display = 'none';
*/
