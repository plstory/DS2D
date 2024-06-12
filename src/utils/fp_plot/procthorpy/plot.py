from matplotlib import patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from .utils import ROOM_COLOR, ROOM_TYPE


def floorplan_to_color(data):
    room_colors = []
    for room in data:
        room_type_key = next(key for key, value in ROOM_TYPE.items() if value == room['room_type'])
        color = ROOM_COLOR[room_type_key]
        room_colors.append((room['floor_polygon'], color, room['room_type']))
    return room_colors


def plot_floorplan(data, ax=None, title=None, wall_thickness=0.4, label_rooms=False):
    room_colors = floorplan_to_color(data)
        
    for polygon, color, room_type in room_colors:
        polygon_points = [(point['x'], point['z']) for point in polygon]
        color_normalized = [c / 255.0 for c in color]
        
        # Draw room
        polygon_shape = patches.Polygon(polygon_points, closed=True, edgecolor='black', facecolor=color_normalized, linewidth=2)
        ax.add_patch(polygon_shape)
        
        # # Draw walls
        for i in range(len(polygon_points)):
            start_point = polygon_points[i]
            end_point = polygon_points[(i + 1) % len(polygon_points)]
            ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color=[c / 255.0 for c in ROOM_COLOR[14]], linewidth=wall_thickness * 10)

        # Room label
        if label_rooms:
            centroid = np.mean(polygon_points, axis=0)
            ax.text(centroid[0], centroid[1], room_type, ha='center', va='center', fontsize=6, weight='bold', color='black')
    
    ax.set_xlim(-1, max(p['x'] for room in data for p in room['floor_polygon']) + 1)
    ax.set_ylim(-1, max(p['z'] for room in data for p in room['floor_polygon']) + 1)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    
    if title:
        ax.set_title(title)
    
    return ax
