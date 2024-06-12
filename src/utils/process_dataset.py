import os
from .json_repair import *
from .polygon_object import Polygon
from .json_check import is_valid_json
import math
import networkx as nx
from skimage.draw import polygon_perimeter, set_color
from skimage.draw import polygon as draw_polygon

from skimage import io

class Floorplan():
    def __init__(self, path, json_str=None):
        self.path = path
        if json_str is None:
            self.floorplan = json_loads(json_load(open(self.path, 'r')))
        else:
            self.floorplan = json_loads(json_str)
        self.validate_normal = is_valid_json(self.floorplan)
        self.validate_strict = is_valid_json(self.floorplan, strict=True)
        self.room_infos = {}

    def get_room_count(self):
        return self.floorplan["room_count"]

    
    def get_total_area(self):
        return self.floorplan["total_area"]
    
    def get_room_types(self):
        try:
            return self.floorplan["room_types"]
        except KeyError:
            self.floorplan["room_types"] = []
            for room in self.get_rooms():
                try:
                    self.floorplan["room_types"].append(room["room_type"])
                except KeyError:
                    pass
            return self.floorplan["room_types"]
    
    def get_rooms(self):
        try:
            return self.floorplan["rooms"]
        except KeyError:
            return []
    
    def get_num_rooms(self):
        return len(self.get_rooms())
    
    def get_room_ids(self):
        ret = []
        for i, room in enumerate(self.get_rooms()):
            try:
                ret.append(room["id"])
            except KeyError:
                room["id"] = f"Placeholder|{i}"
        return ret
    
    def get_unmodified_room_ids(self):
        ret = set()
        for room in self.get_rooms():
            try:
                room_id = room["id"]
                if "Placeholder" not in room_id:
                    ret.add(room_id)
            except KeyError:
                pass
        return ret
    
    def get_edges(self):
        return self.floorplan["edges"]
    

    def get_room(self, room_id):
        assert self.check_room_exists(room_id), f"Room with id {room_id} does not exist"
        for room in self.get_rooms():
            if room["id"] == room_id: # always return the first room with the same id
                return room

    def check_room_exists(self, room_id):
        return room_id in self.get_room_ids()
    
    def count_room_overlaps(self):
        buff = 0
        room_ids = self.get_room_ids()
        for i in range(len(room_ids)-1):
            for j in range(i+1, len(room_ids)):
                room1_id, room2_id = room_ids[i], room_ids[j]
                if room1_id == room2_id: # just in case if there are rooms with the same id
                    continue
                try:
                    if self.get_room_polygon(room1_id).overlap(self.get_room_polygon(room2_id)):
                        buff += 1
                except:
                    pass
        return buff
    
    def get_room_area(self, room_id):
        assert self.check_room_exists(room_id), f"Room with id {room_id} does not exist"
        try:
            room = self.get_room(room_id)
            return room["area"]
        except Exception:
            try:
                room["area"] = self.compute_room_area(room_id)
                return room["area"]
            except Exception:
                room["area"] = None
                return None
    
    def get_room_polygon(self, room_id):
        assert self.check_room_exists(room_id), f"Room with id {room_id} does not exist"
        room = self.get_room(room_id)
        try:
            return room["polygon_object"]
        except KeyError:
            try:
                room["polygon_object"] = Polygon(room["floor_polygon"])
            except:
                room["polygon_object"] = None
            return room["polygon_object"]

    def get_room_polygon_area(self, room_id):
        assert self.check_room_exists(room_id), f"Room with id {room_id} does not exist"
        room = self.get_room(room_id)
        try:
            return room['polygon_area']
        except KeyError:
            try:
                room_polygon = self.get_room_polygon(room_id)
                room['polygon_area'] = room_polygon.sorted_area, room_polygon.sorted_area == room_polygon.unsorted_area
            except:
                room['polygon_area'] = None, None
                return room['polygon_area']
            
            return room['polygon_area']

    
    def get_room_type(self, room_id):
        assert self.check_room_exists(room_id), f"Room with id {room_id} does not exist"
        room = self.get_room(room_id)
        return room["room_type"]
    
    def get_room_height(self, room_id):
        assert self.check_room_exists(room_id), f"Room with id {room_id} does not exist"
        room = self.get_room(room_id)
        return room["height"]
    
    def get_room_width(self, room_id):
        assert self.check_room_exists(room_id), f"Room with id {room_id} does not exist"
        room = self.get_room(room_id)
        return room["width"]
    
    '''
        graph extraction section below
    '''
    
    def extract_polygon(self):
        room_info = self.extract_room_info()
        return [room['floor_polygon'] for room in room_info]
    
    def extract_room_info(self, json_str = None):
        if json_str is None:
            rooms = self.get_rooms()
        else:
            json_d = json_loads(json_str)
            rooms = json_d['rooms']
        rooms_info = []
        for room in rooms:
            room_d = {}
            try:
                vertices = room['floor_polygon']
            except:
                continue
            polygon = []
            for vertix in vertices:
                if 'x' in vertix.keys() and 'z' in vertix.keys():
                    polygon.append([vertix['x'],vertix['z']])
            room_d['floor_polygon'] = polygon
            if 'room_type' in room.keys():
                room_d['room_type'] = room['room_type']
            if 'id' in room.keys():
                room_d['id'] = room['id']
            rooms_info.append(room_d)
        return rooms_info

    def polygon2bbox(self, polygon):
        x_max, x_min, y_max, y_min = 0, math.inf, 0, math.inf
        for x,y in polygon:
            x_max = max(x_max,x)
            x_min = min(x_min,x)
            y_max = max(y_max,y)
            y_min = min(y_min,y)
        return (x_min, y_min, x_max, y_max)

    def bboxes2bubble(self, bboxes, th=9):
        '''
            bboxes: list of xyxy definitions for each room
        '''
        edges = []
        for u in range(len(bboxes)):
            for v in range(u+1,len(bboxes)):
                if not self.collide2d(bboxes[u][:4],bboxes[v][:4],th=th): 
                    continue
                edges.append((u,v))
                
        return edges

    def collide2d(self, bbox1, bbox2, th=0):
        return not(
            (bbox1[0]-th > bbox2[2]) or
            (bbox1[2]+th < bbox2[0]) or
            (bbox1[1]-th > bbox2[3]) or
            (bbox1[3]+th < bbox2[1])
        )

    def draw_polygons_from_str(self, json_str):
        room_info = self.extract_room_info(json_str=json_str)
        self.draw_polygons_ext(room_info)


    def draw_polygons_ext(self, room_info=None):
        import numpy as np
        if room_info is None:
            room_info = self.extract_room_info()
        polygons = [room['floor_polygon'] for room in room_info]
        room_types = [room['room_type'].lower() for room in room_info]
        r2c = {'livingroom':[255,0,0],
               'kitchen':[0,255,0],
               'bathroom':[0,0,255],
               'bedroom':[0,255,255]
               }
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        
        for idx, polygon in enumerate(polygons):
            room_type = room_types[idx]
            xs = [10*p[0] for p in polygon]
            ys = [10*p[1] for p in polygon]
            rr, cc = draw_polygon(xs,ys)
            set_color(img,[rr,cc],r2c[room_type])
            rr, cc = polygon_perimeter(xs,ys)
            set_color(img,[rr,cc],[0,0,0])
            io.imsave('tmp.png',img)
        io.imsave('tmp.png',img)
    
    def draw_polygons(self, rplan=False):
        import numpy as np
        ids = self.get_room_ids()
        r2c = {'livingroom':[255,0,0],
               'kitchen':[0,255,0],
               'bathroom':[0,0,255],
               'bedroom':[0,255,255],
               'masterroom':[255,0,255],
               'diningroom':[255,255,0],
               'childroom':[0,0,0],
               'studyroom':[127,255,127],
               'secondroom':[127,127,255],
               'guestroom':[255,127,127],
               'balcony':[0,127,255],
               'entrance':[0,255,127],
               'storage':[127,127,0],
               'wall-in':[127,0,127],
               'storage':[0,127,127],
               }
        if rplan:
            factor = 1
        else:
            factor = 10
        img1 = np.zeros((300, 300, 3), dtype=np.uint8)
        img2 = np.zeros((300, 300, 3), dtype=np.uint8)
        room_types = self.get_room_types()
        polygons = []
        for idx, id in enumerate(ids):
            room_type = room_types[idx].lower()
            polygon = self.get_room_polygon(id)
            polygons.append(polygon)
            plg1 = [[d['x'],d['z']] for  d in polygon.vertices]
            xs = [factor*p[0] for p in plg1]
            ys = [factor*p[1] for p in plg1]
            rr, cc = draw_polygon(xs,ys)
            set_color(img1,[rr,cc],r2c[room_type])
            rr, cc = polygon_perimeter(xs,ys)
            set_color(img1,[rr,cc],[0,0,0])
            # io.imsave('tmp_unsort.png',img1)

            plg2 = [[d['x'],d['z']] for  d in polygon.sorted_vertices]
            xs = [10*p[0] for p in plg2]
            ys = [10*p[1] for p in plg2]
            rr, cc = draw_polygon(xs,ys)
            set_color(img2,[rr,cc],r2c[room_type])
            rr, cc = polygon_perimeter(xs,ys)
            set_color(img2,[rr,cc],[0,0,0])
            # io.imsave('tmp_sorted.png',img2
            # import pdb;pdb.set_trace()
            
            
        io.imsave('tmp_unsort.png',img1)
        io.imsave('tmp_sorted.png',img2)

    def get_graph_from_polygons(self, dataset='rplan'):
        rplan = dataset == 'rplan'
        # self.draw_polygons(rplan=rplan)
        room_info = self.extract_room_info()
        try:
            polygons = [room['floor_polygon'] for room in room_info]
            bboxes = [self.polygon2bbox(pg) for pg in polygons]
            if dataset == 'rplan':
                th = 9
            else:
                th = 2
            edges = self.bboxes2bubble(bboxes,th=th)
            G = nx.Graph()
            for idx, room_d in enumerate(room_info):
                if 'room_type' not in room_d.keys() or 'id' not in room_d.keys():
                    return None
                G.add_node(idx, room_type=room_d['room_type'], id=room_d['id'])
            G.add_edges_from(edges)
        except:
            return None
        return G


class FloorplansAndPrompt():

    def __init__(self, path, json_strs = None):
        self.path = path
        from ..utils import list_json_files
        self.floorplans = {}
        if json_strs is None:
            for plan in list_json_files(self.path):
                self.floorplans[plan[:-5]] = Floorplan(os.path.join(self.path, plan))
        else:
            for idx , json_str in enumerate(json_strs):
                self.floorplans[idx] = Floorplan(path=None, json_str=json_str)

def is_same_node(n1,n2):
    is_same_id = str(n1['id']).lower() == str(n2['id']).lower() 
    is_same_type = n1['room_type'].lower() == n2['room_type'].lower()
    return is_same_type