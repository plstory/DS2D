import json
from ..utils import repair_json

def extract_output_json(input_str: str):
    try:
        output_index = input_str.find("Output:")
        if output_index == -1:
            output_index = input_str.find("assistant") + len("assistant")
        else:
           output_index + len("Output:\n")
        if output_index == -1:
            return None

        output_str = input_str[output_index:]
        output_dict = json.loads(output_str)

        return output_dict
    except json.JSONDecodeError:
        try:
            json_repaired = repair_json(output_str, return_objects=True)
            if json_repaired != "":
                return json_repaired
            else:
                return {}
        except Exception:
            return {}
