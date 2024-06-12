from skimage import io
from skimage import morphology,feature,transform,measure
from skimage.draw import line
from pathlib import Path
import math
from scipy import stats
from scipy import ndimage
from shapely import geometry
import numpy as np
import pickle
from datasets import load_from_disk
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import scipy.io as sio
import pickle

def isEdge(img, x, y, xp, yp, c1, c2, edges, vertices):
    if (x,y) in edges:
        return
    if img[x,y,1] != c1 or img[x,y,2] != c2:
        return 
    isvertix = sum([(img[x+1,y,1:3] == (c1,c2)).all(), 
                    (img[x-1,y,1:3] == (c1,c2)).all(), 
                    (img[x,y+1,1:3] == (c1,c2)).all(), 
                    (img[x,y-1,1:3] == (c1,c2)).all(), 
                    (img[x+1,y+1,1:3] == (c1,c2)).all(), 
                    (img[x-1,y-1,1:3] == (c1,c2)).all(), 
                    (img[x+1,y-1,1:3] == (c1,c2)).all(), 
                    (img[x-1,y+1,1:3] == (c1,c2)).all()])
    isedge = isvertix < 8
    isvertix = (isvertix == 3) or (isvertix==7)
    if isvertix:
        vertices.append((x,y))
    if isedge:
        edges.add((x,y))
        for xn,yn in [(x+1,y),(x-1,y,),(x,y+1),(x,y-1)]:
            if (xn,yn) != (xp,yp):
                isEdge(img,xn,yn,x,y,c1,c2,edges,vertices)
        # isEdge(img,x-1,y,c1,c2,edges,vertices)
        # isEdge(img,x,y+1,c1,c2,edges,vertices)
        # isEdge(img,x,y-1,c1,c2,edges,vertices)

room_label = [(0, 'LivingRoom', 1, "PublicArea"),
            (1, 'MasterRoom', 0, "Bedroom"),
            (2, 'Kitchen', 1, "FunctionArea"),
            (3, 'Bathroom', 0, "FunctionArea"),
            (4, 'DiningRoom', 1, "FunctionArea"),
            (5, 'ChildRoom', 0, "Bedroom"),
            (6, 'StudyRoom', 0, "Bedroom"),
            (7, 'SecondRoom', 0, "Bedroom"),
            (8, 'GuestRoom', 0, "Bedroom"),
            (9, 'Balcony', 1, "PublicArea"),
            (10, 'Entrance', 1, "PublicArea"),
            (11, 'Storage', 0, "PublicArea"),
            (12, 'Wall-in', 0, "PublicArea"),
            (13, 'External', 0, "External"),
            (14, 'ExteriorWall', 0, "ExteriorWall"),
            (15, 'FrontDoor', 0, "FrontDoor"),
            (16, 'InteriorWall', 0, "InteriorWall"),
            (17, 'InteriorDoor', 0, "InteriorDoor")]
    
def savemat(file_path,data):
    sio.savemat(file_path,data)

def loadmat(file_path):
    return sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)

def savepkl(file_path,data):
    pickle.dump(data,open(file_path,'wb'))

def loadpkl(file_path):
    return pickle.load(open(file_path,'rb'))

def get_color_map():
    color = np.array([
        [244,242,229], # living room
        [253,244,171], # bedroom
        [234,216,214], # kitchen
        [205,233,252], # bathroom
        [208,216,135], # balcony
        [185,231,168], # balcony
        [249,222,189], # Storage
        [ 79, 79, 79], # exterior wall
        [255,225, 25], # FrontDoor
        [128,128,128], # interior wall
        [255,255,255]
    ],dtype=np.int64)
    cIdx  = np.array([1,2,3,4,1,2,2,2,2,5,1,6,1,10,7,8,9,10])-1
    return color[cIdx]

def collide2d(bbox1, bbox2, th=0):
    return not(
        (bbox1[0]-th > bbox2[2]) or
        (bbox1[2]+th < bbox2[0]) or
        (bbox1[1]-th > bbox2[3]) or
        (bbox1[3]+th < bbox2[1])
    )

edge_type = ['left-above',
    'left-below',
    'left-of',
    'above',
    'inside',
    'surrounding',
    'below',
    'right-of',
    'right-above',
    'right-below']

def point_box_relation(u,vbox):
    uy,ux = u
    vy0, vx0, vy1, vx1 = vbox
    if (ux<vx0 and uy<=vy0) or (ux==vx0 and uy==vy0):
        relation = 0 # 'left-above'
    elif (vx0<=ux<vx1 and uy<=vy0):
        relation = 3 # 'above'
    elif (vx1<=ux and uy<vy0) or (ux==vx1 and uy==vy0):
        relation = 8 # 'right-above'
    elif (vx1<=ux and vy0<=uy<vy1):
        relation = 7 # 'right-of'
    elif (vx1<ux and vy1<=uy) or (ux==vx1 and uy==vy1):
        relation = 9 # 'right-below'
    elif (vx0<ux<=vx1 and vy1<=uy):
        relation = 6 # 'below'
    elif (ux<=vx0 and vy1<uy) or (ux==vx0 and uy==vy1):
        relation = 1 # 'left-below'
    elif(ux<=vx0 and vy0<uy<=vy1):
        relation = 2 # 'left-of'
    elif(vx0<ux<vx1 and vy0<uy<vy1):
        relation = 4 # 'inside'

    return relation

def get_edges(boxes,th=9):
    edges = []
    for u in range(len(boxes)):
        for v in range(u+1,len(boxes)):
            if not collide2d(boxes[u,:4],boxes[v,:4],th=th): continue
            uy0, ux0, uy1, ux1 = boxes[u,:4].astype(int)
            vy0, vx0, vy1, vx1 = boxes[v,:4].astype(int)
            uc = (uy0+uy1)/2,(ux0+ux1)/2
            vc = (vy0+vy1)/2,(vx0+vx1)/2
            if ux0 < vx0 and ux1 > vx1 and uy0 < vy0 and uy1 > vy1:
                relation = 5 #'surrounding'
            elif ux0 >= vx0 and ux1 <= vx1 and uy0 >= vy0 and uy1 <= vy1:
                relation = 4 #'inside'
            else:
                relation = point_box_relation(uc,boxes[v,:4])
            edges.append([u,v,relation])
            
    edges = np.array(edges,dtype=int)
    return edges

door_pos = [
    'nan',
    'bottom',
    'bottom-right','right-bottom',
    'right',
    'right-top','top-right',
    'top',
    'top-left','left-top',
    'left',
    'left-bottom','bottom-left'
]

def door_room_relation(d_center,r_box):
    y0,x0,y1,x1 = r_box
    yc,xc = (y1+y0)/2, (x0+x1)/2
    y,x = d_center
    
    if x==xc and y<yc:return 7
    elif x==xc and y>yc:return 1
    elif y==yc and x<xc:return 10
    elif y==yc and x>xc:return 4
    elif x0<x<xc:
        if y<yc:return 8
        else:return 12
    elif xc<x<x1:
        if y<yc:return 6
        else:return 2
    elif y0<y<yc:
        if x<xc:return 9
        else:return 5
    elif yc<y<y1:
        if x<xc:return 11
        else:return 3
    else:return 0

class Floorplan():

    @property
    def boundary(self): return self.image[...,0]
    
    @property
    def category(self): return self.image[...,1]

    @property
    def instance(self): return self.image[...,2]
    
    @property
    def inside(self): return self.image[...,3]

    def __init__(self,file_path):
        self.path = file_path
        self.name = Path(self.path).stem
        self.image = io.imread(self.path)
        self.h,self.w,self.c = self.image.shape
        self.corrupted = False
        
        self.front_door = None
        self.exterior_boundary = None
        self.rooms = None
        self.edges = None

        self.archs = None
        self.graph = None

        # self._get_front_door()
        # self._get_exterior_boundary()
        self._get_rooms()
        self._get_edges()
        
    def __repr__(self): 
        return f'{self.name},({self.h},{self.w},{self.c})'

    def _get_front_door(self):
        front_door_mask = self.boundary==255
        # fast bbox
        # min_h,max_h = np.where(np.any(front_door_mask,axis=1))[0][[0,-1]]
        # min_w,max_w = np.where(np.any(front_door_mask,axis=0))[0][[0,-1]]  
        # self.front_door = np.array([min_h,min_w,max_h,max_w],dtype=int)
        region = measure.regionprops(front_door_mask.astype(int))[0]
        self.front_door = np.array(region.bbox,dtype=int)

    def _get_exterior_boundary(self):
        if self.front_door is None: self._get_front_door()
        self.exterior_boundary = []

        min_h,max_h = np.where(np.any(self.boundary,axis=1))[0][[0,-1]]
        min_w,max_w = np.where(np.any(self.boundary,axis=0))[0][[0,-1]]
        min_h = max(min_h-10,0)
        min_w = max(min_w-10,0)
        max_h = min(max_h+10,self.h)
        max_w = min(max_w+10,self.w)

        # src: http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html
        # search direction:0(right)/1(down)/2(left)/3(up)
        # find the left-top point
        flag = False
        for h in range(min_h, max_h):
            for w in range(min_w, max_w):
                if self.inside[h, w] == 255:
                    self.exterior_boundary.append((h, w, 0))
                    flag = True
                    break
            if flag:
                break
        
        # left/top edge: inside
        # right/bottom edge: outside
        while(flag):
            if self.exterior_boundary[-1][2] == 0:
                for w in range(self.exterior_boundary[-1][1]+1, max_w):
                    corner_sum = 0
                    if self.inside[self.exterior_boundary[-1][0], w] == 255:
                        corner_sum += 1
                    if self.inside[self.exterior_boundary[-1][0]-1, w] == 255:
                        corner_sum += 1
                    if self.inside[self.exterior_boundary[-1][0], w-1] == 255:
                        corner_sum += 1
                    if self.inside[self.exterior_boundary[-1][0]-1, w-1] == 255:
                        corner_sum += 1
                    if corner_sum == 1:
                        new_point = (self.exterior_boundary[-1][0], w, 1)
                        break
                    if corner_sum == 3:
                        new_point = (self.exterior_boundary[-1][0], w, 3)
                        break
            
            if self.exterior_boundary[-1][2] == 1:      
                for h in range(self.exterior_boundary[-1][0]+1, max_h): 
                    corner_sum = 0                
                    if self.inside[h, self.exterior_boundary[-1][1]] == 255:
                        corner_sum += 1
                    if self.inside[h-1, self.exterior_boundary[-1][1]] == 255:
                        corner_sum += 1
                    if self.inside[h, self.exterior_boundary[-1][1]-1] == 255:
                        corner_sum += 1
                    if self.inside[h-1, self.exterior_boundary[-1][1]-1] == 255:
                        corner_sum += 1
                    if corner_sum == 1:
                        new_point = (h, self.exterior_boundary[-1][1], 2)
                        break
                    if corner_sum == 3:
                        new_point = (h, self.exterior_boundary[-1][1], 0)
                        break

            if self.exterior_boundary[-1][2] == 2:   
                for w in range(self.exterior_boundary[-1][1]-1, min_w, -1):
                    corner_sum = 0                     
                    if self.inside[self.exterior_boundary[-1][0], w] == 255:
                        corner_sum += 1
                    if self.inside[self.exterior_boundary[-1][0]-1, w] == 255:
                        corner_sum += 1
                    if self.inside[self.exterior_boundary[-1][0], w-1] == 255:
                        corner_sum += 1
                    if self.inside[self.exterior_boundary[-1][0]-1, w-1] == 255:
                        corner_sum += 1
                    if corner_sum == 1:
                        new_point = (self.exterior_boundary[-1][0], w, 3)
                        break
                    if corner_sum == 3:
                        new_point = (self.exterior_boundary[-1][0], w, 1)
                        break

            if self.exterior_boundary[-1][2] == 3:       
                for h in range(self.exterior_boundary[-1][0]-1, min_h, -1):
                    corner_sum = 0                
                    if self.inside[h, self.exterior_boundary[-1][1]] == 255:
                        corner_sum += 1
                    if self.inside[h-1, self.exterior_boundary[-1][1]] == 255:
                        corner_sum += 1
                    if self.inside[h, self.exterior_boundary[-1][1]-1] == 255:
                        corner_sum += 1
                    if self.inside[h-1, self.exterior_boundary[-1][1]-1] == 255:
                        corner_sum += 1
                    if corner_sum == 1:
                        new_point = (h, self.exterior_boundary[-1][1], 0)
                        break
                    if corner_sum == 3:
                        new_point = (h, self.exterior_boundary[-1][1], 2)
                        break

            if new_point != self.exterior_boundary[0]:
                self.exterior_boundary.append(new_point)
            else:
                flag = False
        self.exterior_boundary = [[r,c,d,0] for r,c,d in self.exterior_boundary]
        
        door_y1,door_x1,door_y2,door_x2 = self.front_door
        door_h,door_w = door_y2-door_y1,door_x2-door_x1
        is_vertical = door_h>door_w or door_h==1 # 

        insert_index = None
        door_index = None
        new_p = []
        th = 3
        for i in range(len(self.exterior_boundary)):
            y1,x1,d,_ = self.exterior_boundary[i]
            y2,x2,_,_ = self.exterior_boundary[(i+1)%len(self.exterior_boundary)] 
            if is_vertical!=d%2: continue
            if is_vertical and (x1-th<door_x1<x1+th or x1-th<door_x2<x1+th): # 1:down 3:up
                l1 = geometry.LineString([[y1,x1],[y2,x2]])    
                l2 = geometry.LineString([[door_y1,x1],[door_y2,x1]])  
                l12 = l1.intersection(l2)
                if l12.length>0:
                    dy1,dy2 = l12.xy[0] # (y1>y2)==(dy1>dy2)
                    insert_index = i
                    door_index = i+(y1!=dy1)
                    if y1!=dy1: new_p.append([dy1,x1,d,1])
                    if y2!=dy2: new_p.append([dy2,x1,d,1])
            elif not is_vertical and (y1-th<door_y1<y1+th or y1-th<door_y2<y1+th):
                l1 = geometry.LineString([[y1,x1],[y2,x2]])    
                l2 = geometry.LineString([[y1,door_x1],[y1,door_x2]])  
                l12 = l1.intersection(l2)
                if l12.length>0:
                    dx1,dx2 = l12.xy[1] # (x1>x2)==(dx1>dx2)
                    insert_index = i
                    door_index = i+(x1!=dx1)
                    if x1!=dx1: new_p.append([y1,dx1,d,1])
                    if x2!=dx2: new_p.append([y1,dx2,d,1])                

        if len(new_p)>0:
            self.exterior_boundary = self.exterior_boundary[:insert_index+1]+new_p+self.exterior_boundary[insert_index+1:]
        self.exterior_boundary = self.exterior_boundary[door_index:]+self.exterior_boundary[:door_index]

        self.exterior_boundary = np.array(self.exterior_boundary,dtype=int)

    def _get_rooms(self):
        polygon = []
        rooms = []
        regions = measure.regionprops(self.instance)
        for region in regions:
            c1 = stats.mode(self.category[region.coords[:,0],region.coords[:,1]])[0]
            c2 = stats.mode(self.instance[region.coords[:,0],region.coords[:,1]])[0]
            y0,x0,y1,x1 = np.array(region.bbox) 
            yc, xc = y0, x0
            if not (self.image[y0,x0,1:3] == (c1,c2)).all():
                for x in range(256):
                    if (self.image[yc,x,1:3] == (c1,c2)).all():
                        xc = x
                        break
            es = set()
            vs = []
            if not (self.image[yc,xc+1,1:3] == (c1,c2)).all():
                self.corrupted = True
                break

            isEdge(self.image,yc,xc,yc,xc+1,c1,c2,es,vs)
            if(len(vs)<4):
                self.corrupted = True
                break
            # polygons, height, width = get_sorted_vertices(list(vs))
            polygons = vs
            min_x, max_x = min_y, max_y = np.inf, -np.inf
            for vertex in polygons:
                min_x = min(min_x, vertex[0])
                max_x = max(max_x, vertex[0])
                min_y = min(min_y, vertex[1])
                max_y = max(max_y, vertex[1])
            height = max_y - min_y
            width = max_x - min_x
            area = get_area(polygons)
                # import pdb; pdb.set_trace()
            polygon.append(polygons)
            rooms.append([y0,x0,y1,x1,c1,c2,area, height, width])
        
        self.rooms = np.array(rooms,dtype=int)
        self.polygon = polygon

    def _get_edges(self,th=9):
        if self.rooms is None: self._get_rooms()
        edges = []
        for u in range(len(self.rooms)):
            for v in range(u+1,len(self.rooms)):
                if not collide2d(self.rooms[u,:4],self.rooms[v,:4],th=th): continue
                uy0, ux0, uy1, ux1, c1, _, _ , _, _ = self.rooms[u]
                vy0, vx0, vy1, vx1, c2, _, _ , _, _ = self.rooms[v]
                uc = (uy0+uy1)/2,(ux0+ux1)/2
                vc = (vy0+vy1)/2,(vx0+vx1)/2
                if ux0 < vx0 and ux1 > vx1 and uy0 < vy0 and uy1 > vy1:
                    relation = 5 #'surrounding'
                elif ux0 >= vx0 and ux1 <= vx1 and uy0 >= vy0 and uy1 <= vy1:
                    relation = 4 #'inside'
                else:
                    relation = point_box_relation(uc,self.rooms[v,:4])
                edges.append([u,v,relation])
                
        self.edges = np.array(edges,dtype=int)

    def to_dict(self,xyxy=True,dtype=int):
        '''
        Compress data, notice:
        !!! int->uint8: a(uint8)+b(uint8) may overflow !!!
        '''
        return {
            'name'      :self.name,
            'types'     :self.rooms[:,-1].astype(dtype),
            'boxes'     :(self.rooms[:,[1,0,3,2]]).astype(dtype) 
            if xyxy else self.rooms[:,:4].astype(dtype),
            'boundary'  :self.exterior_boundary[:,[1,0,2,3]].astype(dtype)
            if xyxy else self.exterior_boundary.astype(dtype),
            'edges'     :self.edges.astype(dtype)
        }

def drawbox(img,xyxy):
    x1, y1, x2, y2, = xyxy
    for x in range(x1,x2+1):
        img[x,y1,:] = 255
        img[x,y2,:] = 255
    for y in range(y1,y2+1):
        img[x1,y,:] = 255
        img[x2,y,:] = 255

def debug_img(bbox,coord):
    img = np.zeros((256,256,3)).astype(np.uint8)
    y0, x0, y1, x1 = bbox
    for x,y in coord:
        img[x,y,1] = 255
    img[y0,x0, :] = 0
    img[y0,x0, 0] = 255
    img[y0,x0, 2] = 255
    img[y1,x1, :] = 0
    img[y1,x1, 0] = 255
    io.imsave('tmp.png',img)

def draw_edge(vertices,img=None):
    if img is None:
        img = np.zeros((256,256,3)).astype(np.uint8)
    coords = vertices.copy()
    coords.append(coords[0])
    for idx in range(len(coords)-1):
        x0,y0 = coords[idx]
        x1,y1 = coords[idx+1]
        rr,cc = line(x0,y0,x1,y1)
        img[rr,cc,:] = 255
    io.imsave('plots/tmp.png',img)
    return img



def calculate_polygon_area(polygon, decimals=2): # shoelace formula
    n = len(polygon)
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    area = abs(area) / 2.0
    return round(area, decimals)

def convert2json():
    room_label = {
                0: 'LivingRoom',
                1: 'MasterRoom', 
                2: 'Kitchen', 
                3: 'Bathroom',
                4: 'DiningRoom',
                5: 'ChildRoom',
                6: 'StudyRoom',
                7: 'SecondRoom',
                8: 'GuestRoom', 
                9: 'Balcony',
                10: 'Entrance',
                11: 'Storage', 
                12: 'Wall-in', 
                13: 'External',
                14: 'ExteriorWall',
                15: 'FrontDoor', 
                16: 'InteriorWall',
                17: 'InteriorDoor',
    }
    pixel2len = 18/256
    pixel2area = pixel2len**2
    fpkl = '/network/scratch/l/luozhiha/datasets/rplan_v2_converted.pkl'
    with open(fpkl,'rb') as f:
        all_data = pickle.load(f)
    for data in all_data:
        json_str = data2json(data)


def getstats(dset):
    stats = defaultdict(list)

    for idx, data in enumerate(dset):
        num_room = len(data['rooms'])
        stats[num_room] += [idx]
    room_split = defaultdict(int)
    for key in stats.keys():
        room_split[key] = len(stats[key])
    return room_split,stats

def get_area(polygon): # shoelace formula
    n = len(polygon)
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    area = abs(area) / 2.0
    return area

def get_sorted_vertices(vertices):
    mid_x, mid_y = sum(i for i, _ in vertices)/len(vertices), sum(j for _, j in vertices)/len(vertices)
    def get_slope_from_mid_point(vertices, mid_x, mid_y):
        ret = []
        min_x, max_x = min_y, max_y = np.inf, -np.inf
        for vertex in vertices:
            # math.atan2(y,x)
            min_x = min(min_x, vertex[0])
            max_x = max(max_x, vertex[0])
            min_y = min(min_y, vertex[1])
            max_y = max(max_y, vertex[1])
            ret.append((vertex, math.atan2(vertex[1] - mid_y, vertex[0] - mid_x)))
        height = max_y - min_y
        width = max_x - min_x
        return ret, (height, width)
    def sort_vertices(vertices):
        return sorted(vertices, key=lambda x: x[1])
    buff = get_slope_from_mid_point(vertices, mid_x, mid_y)
    tmp, (height, width) = sort_vertices(buff[0]), buff[1]
    return [v[0] for v in tmp], height, width


def data2json(data):
    room_label = {
        0: "LivingRoom",
        1: "MasterRoom", 
        2: "Kitchen", 
        3: "Bathroom",
        4: "DiningRoom",
        5: "ChildRoom",
        6: "StudyRoom",
        7: "SecondRoom",
        8: "GuestRoom", 
        9: "Balcony",
        10: "Entrance",
        11: "Storage", 
        12: "Wall-in", 
        13: "External",
        14: "ExteriorWall",
        15: "FrontDoor", 
        16: "InteriorWall",
        17: "InteriorDoor",
    }
    pixel2len = 18/256
    pixel2area = pixel2len**2
    
    num_rooms = len(data['rooms'])
    total_area = 0
    room_types = []
    json_str = f'"rooms": ['
    for room_idx, room_info in enumerate(data['rooms']):
        y0,x0,y1,x1,c1,c2,area, height, width = room_info
        total_area += area
        json_str += f'{{"area": {area*pixel2area:.2f}, '
        json_str += f'"room_type": "{room_label[c1]}", '
        room_types.append(room_label[c1])
        json_str += '"floor_polygon": ['
        for x,y in data['polygons'][room_idx]:
            json_str += f'{{"x": {x}, "z": {y}}}, '
        json_str = json_str.strip(', ') + '], '
        json_str += f'"height": {height*pixel2len:.2f}, '
        json_str += f'"width": {width*pixel2len:.2f}, '
        json_str += f'"id": "room|{room_idx}"}}, '
    json_str = json_str.strip(', ') + ']}'
    json_str = f'{{"room_count": {len(data["rooms"])}, "total_area": {total_area*pixel2area:.2f}, "room_types": {room_types}, ' + json_str
    json_str = json_str.strip(', ')
    json_str = json_str.replace("'",'"')
    return json_str


if __name__ == "__main__":
    from datasets import Dataset, DatasetDict
    
    from src.utils.fp_plot import plot_procthor
    import sys
    import os
    from src.utils import json_loads, json_load

    img = np.zeros((256,256,3)).astype(np.uint8)
    rplan_dir = 'datasets/rplan'
    huge_data = []
    corrupt_idx = []
    # for idx in tqdm(range(80788)):
    for idx in tqdm(range(1000)):
        fpath = f'{rplan_dir}/{idx}.png'
        fp = Floorplan(fpath)
        if fp.corrupted:
            corrupt_idx.append(idx)
            print(f"CORRUPTED: {idx}")
            continue
        # data = fp.to_dict()
        data = {'rooms': fp.rooms,
                'polygons': fp.polygon,
                'edges': fp.edges,
                'png_idx': idx}
        huge_data.append(data)

    ds = Dataset.from_list(huge_data)
    ds = ds.map(lambda x: json_loads(data2json(x)))
    ds = ds.remove_columns(['polygons','png_idx'])
    stats, rooms = getstats(ds)
    ds5 = ds.select(rooms[5])
    ds6 = ds.select(rooms[6])
    ds7 = ds.select(rooms[7])
    ds8 = ds.select(rooms[8])
    ds5 = ds5.train_test_split(0.1)
    ds6 = ds6.train_test_split(0.1)
    ds7 = ds7.train_test_split(0.1)
    ds8 = ds8.train_test_split(0.1)
    dd = {'5': ds5,
          '6': ds6,
          '7': ds7,
          '8': ds8,
          }
    dd = DatasetDict(dd)
    dd.save_to_disk('datasets/rplan_converted')