from typing import List, Dict, Tuple
from datasets import DatasetDict, Dataset
import matplotlib.pyplot as plt

class ProcTHORConverter:
    def __init__(self, metric_unit: str = "meters", round_value: int = 1):
        self.metric_unit = metric_unit
        self.round_value = round_value

    def remove_redundant_points(self, polygon: List[Dict[str, float]]) -> List[Dict[str, float]]:
        if len(polygon) < 3:
            return polygon

        def collinear(p1, p2, p3):
            return (p2['x'] - p1['x']) * (p3['z'] - p1['z']) - (p2['z'] - p1['z']) * (p3['x'] - p1['x']) == 0

        cleaned_polygon = [polygon[0]]
        for i in range(1, len(polygon) - 1):
            if not collinear(polygon[i - 1], polygon[i], polygon[i + 1]):
                cleaned_polygon.append(polygon[i])
        cleaned_polygon.append(polygon[-1])

        return [{key: value for key, value in point.items() if key != 'y'} for point in cleaned_polygon]

    def round_polygon(self, polygon: List[Dict[str, float]]) -> List[Dict[str, float]]:
        return [
            {'x': round(point['x'], self.round_value), 'y': round(point['z'], self.round_value)}
            for point in polygon
        ]

    def calculate_polygon_area(self, polygon: List[Dict[str, float]]) -> float:
        n = len(polygon)
        area = 0
        for i in range(n):
            j = (i + 1) % n
            area += polygon[i]['x'] * polygon[j]['z']
            area -= polygon[j]['x'] * polygon[i]['z']
        return round(abs(area) / 2.0, self.round_value)

    def calculate_dimensions(self, polygon: List[Dict[str, float]]):
        x_coords = [p['x'] for p in polygon]
        z_coords = [p['z'] for p in polygon]
        width = max(x_coords) - min(x_coords)
        height = max(z_coords) - min(z_coords)
        return round(width, self.round_value), round(height, self.round_value)

    def is_polygon_rectangular(self, polygon: List[Dict[str, float]]) -> bool:
        return 1 if len(polygon) == 4 else 0

    def format_rooms_type(self, room_list: List[str]) -> str:
        room_count = {}
        for room in room_list:
            if room in room_count:
                room_count[room] += 1
            else:
                room_count[room] = 1

        formatted_rooms = []
        for room, count in room_count.items():
            if count > 1:
                formatted_rooms.append(f"{count} {room}s")
            else:
                formatted_rooms.append(room)

        return ", ".join(formatted_rooms)

    def format_total_area(self, total_area: int) -> int:
        return round(total_area, self.round_value)

    def create_five_level_prompt(self, room_count: int, rooms_type: List[int], total_area: int, rooms: List[Dict[str, float]]) -> List[str]:
        prompts = []
        prompts.append(f"{{'room_count': {room_count}}}")
        prompts.append(f"{{'room_count': {room_count}, 'total_area': {total_area}}}")
        prompts.append(f"{{'room_count': {room_count}, 'total_area': {total_area}, 'room_types': {rooms_type} }}")

        rooms_details_level_4 = [{"room_type": room["room_type"], "area": room["area"]} for room in rooms]
        prompts.append(f"{{'rooms': {rooms_details_level_4}}}")

        rooms_details_level_5 = [{"room_type": room["room_type"], "width": room["width"], "height": room["height"], "is_regular": room["is_regular"]} for room in rooms]
        prompts.append(f"{{'rooms': {rooms_details_level_5}}}")

        return prompts

    def polygon2bbox(self, polygon: List[Dict[str, float]]):
        x_coords = [point['x'] for point in polygon]
        z_coords = [point['y'] for point in polygon]
        x_min, x_max = min(x_coords), max(x_coords)
        z_min, z_max = min(z_coords), max(z_coords)
        return x_min, z_min, x_max, z_max

    def collide2d(self, bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float], threshold: float = 0) -> bool:
        return not (
            (bbox1[0] - threshold > bbox2[2]) or
            (bbox1[2] + threshold < bbox2[0]) or
            (bbox1[1] - threshold > bbox2[3]) or
            (bbox1[3] + threshold < bbox2[1])
        )

    def bboxes2bubble(self, bboxes: List[Tuple[float, float, float, float]], threshold: float = 0) -> List[Tuple[int, int]]:
        edges = []
        for i in range(len(bboxes)):
            for j in range(i + 1, len(bboxes)):
                if self.collide2d(bboxes[i], bboxes[j], threshold):
                    edges.append((i, j))
        return edges

    def generate_edges(self, rooms: List[Dict], threshold: float = 0) -> List[Tuple[int, int]]:
        bboxes = [self.polygon2bbox(room['floor_polygon']) for room in rooms]
        edges = self.bboxes2bubble(bboxes, threshold)
        return edges

    def element_to_absolute(self, element, wall_data):
        wall0_id = element['wall0']
        wall1_id = element['wall1']
        wall0 = next(w for w in wall_data if w['id'] == wall0_id)

        # Extract wall start and end coordinates
        wall_start_x, wall_start_y = wall0['polygon'][0]['x'], wall0['polygon'][0]['z']
        wall_end_x, wall_end_y = wall0['polygon'][1]['x'], wall0['polygon'][1]['z']

        # Calculate the length of the wall
        wall_length = ((wall_end_x - wall_start_x) ** 2 + (wall_end_y - wall_start_y) ** 2) ** 0.5

        # Get the relative element position (holePolygon is relative to the wall)
        hole_start_x = element['holePolygon'][0]['x']
        hole_start_y = element['holePolygon'][0]['y']
        hole_end_x = element['holePolygon'][1]['x']
        hole_end_y = element['holePolygon'][1]['y']

        # Interpolate element position along the wall and determine if the wall is vertical or horizontal
        if wall_start_x == wall_end_x:
            element_start_abs_x = wall_start_x
            element_start_abs_y = wall_start_y + (wall_end_y - wall_start_y) * (hole_start_x / wall_length)
            element_end_abs_x = wall_end_x
            element_end_abs_y = wall_start_y + (wall_end_y - wall_start_y) * (hole_end_x / wall_length)
        elif wall_start_y == wall_end_y:
            element_start_abs_x = wall_start_x + (wall_end_x - wall_start_x) * (hole_start_x / wall_length)
            element_start_abs_y = wall_start_y
            element_end_abs_x = wall_start_x + (wall_end_x - wall_start_x) * (hole_end_x / wall_length)
            element_end_abs_y = wall_end_y

        return {
              'id': element['id'],
              'polygon': [
                  {'x': round(element_start_abs_x, self.round_value), 'y': round(element_start_abs_y, self.round_value)},
                  {'x': round(element_end_abs_x, self.round_value), 'y': round(element_end_abs_y, self.round_value)}
              ]
          }

    def elements_to_absolute(self, elements, walls):
        absolute_positions = []
        for element in elements:
            abs_pos = self.element_to_absolute(element, walls)
            absolute_positions.append(abs_pos)
        return absolute_positions

    def extract_room_data_from_dataset(self, dataset: List[Dict]) -> List[Dict]:
        extracted_room_data = []
        for data_entry in dataset:
            total_area = 0
            rooms_type = []
            edges = []
            if 'rooms' in data_entry:
                rooms = data_entry["rooms"]
                doors = self.elements_to_absolute(data_entry["doors"], data_entry["walls"])
                windows = self.elements_to_absolute(data_entry["windows"], data_entry["walls"])

                cleaned_rooms = [{k: v for k, v in room.items() if k in ["id", "floorPolygon", "roomType"]} for room in rooms]

                for cleaned_room in cleaned_rooms:
                    cleaned_room["room_type"] = cleaned_room.pop("roomType")
                    cleaned_room["area"] = self.calculate_polygon_area(cleaned_room["floorPolygon"])
                    cleaned_room["width"], cleaned_room["height"] = self.calculate_dimensions(cleaned_room["floorPolygon"])
                    cleaned_room["is_regular"] = self.is_polygon_rectangular(cleaned_room["floorPolygon"])
                    cleaned_room["floor_polygon"] = self.round_polygon(self.remove_redundant_points(cleaned_room["floorPolygon"]))
                    del cleaned_room["floorPolygon"]

                    total_area += cleaned_room["area"]
                    rooms_type.append(cleaned_room["room_type"])

                edges = self.generate_edges(cleaned_rooms)

                entry = {
                    "room_count": len(cleaned_rooms),
                    "total_area": self.format_total_area(total_area),
                    "room_types": rooms_type,
                    "rooms": cleaned_rooms,
                    "edges": edges,
                    "doors": doors,
                    "windows": windows,
                    "prompts": self.create_five_level_prompt(len(cleaned_rooms), rooms_type, self.format_total_area(total_area), cleaned_rooms)
                }

                extracted_room_data.append(entry)
        return extracted_room_data

    def create_dataset(self, dataset: Dict[str, List[Dict]]) -> DatasetDict:
      ds_splits = DatasetDict({
          'train': Dataset.from_list(self.extract_room_data_from_dataset(dataset["train"])),
          'test':  Dataset.from_list(self.extract_room_data_from_dataset(dataset["test"])),
          'validation':  Dataset.from_list(self.extract_room_data_from_dataset(dataset["val"]))
      })

      return ds_splits
  
    def __call__(self, dataset: Dict[str, List[Dict]]) -> DatasetDict:
        return self.create_dataset(dataset)

    def plot_floorplan(self, floorplan, colors = None):
        rooms, doors, windows = floorplan["rooms"], floorplan["doors"], floorplan["windows"]

        if colors is None:
          colors = [
              'red', 'green', 'blue', 'purple', 'orange', 'lime', 'teal', 'grey', 'maroon', 'navy', 'olive', 'silver'
          ]
        fig, ax = plt.subplots()

        for index, room in enumerate(rooms):
            x_coords = [point['x'] for point in room['floor_polygon']]
            z_coords = [point['y'] for point in room['floor_polygon']]

            x_coords.append(x_coords[0])
            z_coords.append(z_coords[0])

            ax.plot(x_coords, z_coords, color=colors[index % len(colors)], label=f"{room['room_type']} ({index})")

        for door in doors:
            start_x = door['polygon'][0]['x']
            start_y = door['polygon'][0]['y']
            end_x = door['polygon'][1]['x']
            end_y = door['polygon'][1]['y']
            ax.plot([start_x, end_x], [start_y, end_y], color='brown', linewidth=3)

        for window in windows:
            start_x = window['polygon'][0]['x']
            start_y = window['polygon'][0]['y']
            end_x = window['polygon'][1]['x']
            end_y = window['polygon'][1]['y']
            ax.plot([start_x, end_x], [start_y, end_y], color='cyan', linewidth=3)

        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_title('Floor plan by room type')
        ax.legend()
        plt.show()

# dataset = prior.load_dataset("procthor-10k")
# converter = ProcTHORConverter(round_value=1)
# new_dataset = converter(dataset)
