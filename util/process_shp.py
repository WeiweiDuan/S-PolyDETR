from osgeo import ogr, osr
import os
import gdal
from gdalconst import *
import math


def convert_to_image_coord0(x, y, path): # convert geocoord to image coordinate
    dataset = gdal.Open(path, GA_ReadOnly)
    adfGeoTransform = dataset.GetGeoTransform()

    dfGeoX=float(x)
    dfGeoY =float(y)
    det = adfGeoTransform[1] * adfGeoTransform[5] - adfGeoTransform[2] *adfGeoTransform[4]

    X = ((dfGeoX - adfGeoTransform[0]) * adfGeoTransform[5] - (dfGeoY -
    adfGeoTransform[3]) * adfGeoTransform[2]) / det

    Y = ((dfGeoY - adfGeoTransform[3]) * adfGeoTransform[1] - (dfGeoX -
    adfGeoTransform[0]) * adfGeoTransform[4]) / det
    return [int(Y),int(X)]

def convert_to_image_coord(x, y, path): # convert geocoord to image coordinate
    ds = gdal.Open(path, GA_ReadOnly )
    target = osr.SpatialReference(wkt=ds.GetProjection())

    source = osr.SpatialReference()
    source.ImportFromEPSG(4269)

    transform = osr.CoordinateTransformation(source, target)

    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(x, y)
    point.Transform(transform)

    x, y = convert_to_image_coord0(point.GetX(), point.GetY(), path)
    return [x, y]


def read_shp(shp_path, tif_path):
    ds = ogr.Open(shp_path)
    layer = ds.GetLayer(0)
    f = layer.GetNextFeature()
    polyline_list = []
    count = 0
    while f:
        geom = f.GetGeometryRef()
        if geom != None:
        # points = geom.GetPoints()
            points = geom.ExportToJson()
            points = eval(points)
            polyline = []
            if points['type'] == "MultiLineString":
                for i in points["coordinates"]:
                    for j in i:
                        tmpt = j
                        if 'sanborn' in shp_path:
                            p = convert_to_image_coord0(tmpt[0], tmpt[1], tif_path)
                        else:
                            p = convert_to_image_coord(tmpt[0], tmpt[1], tif_path)
                        polyline.append([int(p[0]), int(p[1])])
            elif points['type'] ==  "LineString":
                for i in points['coordinates']:
                    tmpt = i
                    if 'sanborn' in shp_path:
                        p = convert_to_image_coord0(tmpt[0], tmpt[1], tif_path)
                    else:
                        p = convert_to_image_coord(tmpt[0], tmpt[1], tif_path) #bray wt use convert_to_image_coord0, shp_name CA_Bray_waterlines_2001_perfect.shp
                    polyline.append([int(p[0]), int(p[1])])

        count += 1
        polyline_list.append(polyline)
        f = layer.GetNextFeature()
    return polyline_list

def interpolation(start, end, inter_dis):
    dis = math.sqrt((start[0]-end[0])**2+(start[1]-end[1])**2)
    segment = []
    if dis == 0:
        return None
    elif dis <= inter_dis:
        return [start, end]
    else:
        ##### calculate k & b in y=kx+b
        add_num = round(dis/inter_dis, 0)   
        segment.append(start)
        if abs(end[1]-start[1]) < 5: ##### vertical line
#             if start == [12146, 2356]:
#                 print('vertical line')
            y_interval = int(round((end[0]-start[0])/float(add_num)))
            for i in range(1, int(add_num)):
                segment.append([start[0]+i*y_interval, start[1]])
        elif abs(end[0]-start[0]) < 5: ##### horizontal line
#             if start == [12146, 2356]:
#                 print('horizontal line: ', end[0], start[0])
            x_interval = int(round((end[1]-start[1])/float(add_num)))
            for i in range(1, int(add_num)):
                segment.append([start[0], start[1]+i*x_interval])
        else:
            k = (end[1]-start[1]) / float(end[0]-start[0])
            b = end[1] - k*end[0]
            x_interval = int(round((end[0]-start[0])/float(add_num)))
            for i in range(1, int(add_num)):
                new_x = start[0]+i*x_interval
                segment.append([int(new_x), int(k*new_x+b)])   
        if end != start:
            segment.append(end)

        return segment
