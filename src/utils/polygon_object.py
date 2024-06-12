import math

def line_intersect(vertex1, vertex2, vertex3, vertex4):
    # check if two lines intersect
    def ccw(A, B, C):
        return (C['z']-A['z']) * (B['x']-A['x']) >= (B['z']-A['z']) * (C['x']-A['x'])
    A, B, C, D = vertex1, vertex2, vertex3, vertex4
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

# def line_intersect_dumb(p1,p2,q1,q2):
#     xp_min = min(p1['x'],p2['x'])
#     xp_max = max(p1['x'],p2['x'])
#     xq_min = min(q1['x'],q2['x'])
#     xq_max = max(q1['x'],q2['x'])
#     zp_min = min(p1['z'],p2['z'])
#     zp_max = max(p1['z'],p2['z'])
#     zq_min = min(q1['z'],q2['z'])
#     zq_max = max(q1['z'],q2['z'])

#     x0_min = max(xp_min,xq_min)
#     x0_max = min(xp_max,xq_max)

#     if x0_max <= x0_min:
#         return False
    
#     z0_min = max(zp_min,zq_min)
#     z0_max = min(zp_max,zq_max)

#     if z0_max <= z0_min:
#         return False

#     rise1 = p2['z'] - p1['z']
#     rise2 = q2['z'] - q1['z']
#     run1 = p2['x'] - p1['z']
#     run2 = q2['x'] - q1['z']

#     if run1 == 0 and run2 ==0:
#         return False

#     m1, m2 = None, None
#     if run1 != 0:
#         m1 = rise1/run1
#     if run2 != 0:
#         m2 = rise2/run2
#     if m1==m2:
#         return False
    
class Polygon:
    def __init__(self, vertices, scaling_factor=18/256):
        self.scaling_factor = scaling_factor
        self.set_vertices(vertices)
        self.edges = self.get_edges()
        self.unsorted_area = self.calculate_polygon_area(self.vertices)
        self.sorted_area = self.calculate_polygon_area(self.sorted_vertices)
        self.width = self.max_x - self.min_x
        self.height = self.max_y - self.min_y
    
    def set_vertices(self, vertices):
        for vertex in vertices:
            vertex['x'] *= self.scaling_factor
            vertex['z'] *= self.scaling_factor
        self.vertices = vertices
        self.sorted_vertices = self.get_sorted_vertices()
    
    def get_edges(self):
        """ Generate edges by creating pairs of points """
        return [(self.vertices[i], self.vertices[(i + 1) % len(self.vertices)]) for i in range(len(self.vertices))]
        # return [(self.sorted_vertices[i], self.sorted_vertices[(i + 1) % len(self.sorted_vertices)]) for i in range(len(self.sorted_vertices))]
    
    def get_sorted_vertices(self):
        def get_midpoint(vertices):
            sum_x, sum_z = 0, 0
            min_x, max_x, min_y, max_y = float('inf'), -float('inf'), float('inf'), -float('inf')
            for vertex in vertices:
                sum_x += vertex['x']
                sum_z += vertex['z']
                min_x, max_x = min(min_x, vertex['x']), max(max_x, vertex['x'])
                min_y, max_y = min(min_y, vertex['z']), max(max_y, vertex['z'])
            return (sum_x/len(vertices), sum_z/len(vertices)), (min_x, max_x, min_y, max_y)
        
        def get_slope_from_mid_point(vertices):
            (mid_x, mid_z), (min_x, max_x, min_y, max_y) = get_midpoint(vertices)
            ret = []
            for vertex in vertices:
                ret.append((vertex, math.atan2(vertex['x'] - mid_x, vertex['z'] - mid_z)))
            return ret, (min_x, max_x, min_y, max_y)
        
        vertices_with_slopes, (self.min_x, self.max_x, self.min_y, self.max_y) = get_slope_from_mid_point(self.vertices)
        vertices_with_slopes = sorted(vertices_with_slopes, key=lambda x: x[1])
        return [vertex[0] for vertex in vertices_with_slopes]

    def calculate_polygon_area(self, vertices, decimals=1): # shoelace formula
        n = len(vertices)
        area = 0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i]['x'] * vertices[j]['z']
            area -= vertices[j]['x'] * vertices[i]['z']
        area = abs(area) / 2.0
        return round(area, decimals)
    
    def surround(self, other):
        # TODO
        pass 
    
    def overlap(self, other):

        for edge1 in self.edges:
            for edge2 in other.edges:
                if line_intersect(edge1[0], edge1[1], edge2[0], edge2[1]):
                    return True
        return False
        

