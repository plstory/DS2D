strict_schema = {
  "type": "object",
  "properties": {
    "room_count": {
      "type": "integer"
    },
    "total_area": {
      "type": "number"
    },
    "room_types": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "rooms": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {
            "type": "string"
          },
          "room_type": {
            "type": "string"
          },
          "area": {
            "type": "number"
          },
          "width": {
            "type": "number"
          },
          "height": {
            "type": "number"
          },
          "is_regular": {
            "type": "integer"
          },
          "floor_polygon": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "x": {
                  "type": "number"
                },
                "z": {
                  "type": "number"
                }
              },
              "required": ["x", "z"]
            }
          }
        },
        "required": ["id", "room_type", "area", "width", "height", "is_regular", "floor_polygon"]
      }
    },
  },
  "required": ["room_count", "total_area", "room_types", "rooms"]
}

schema = {
  "type": "object",
  "properties": {
    "room_count": {
      "type": "integer"
    },
    "total_area": {
      "type": "number"
    },
    "room_types": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "rooms": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {
            "type": "string"
          },
          "room_type": {
            "type": "string"
          },
          "area": {
            "type": "number"
          },
          "width": {
            "type": "number"
          },
          "height": {
            "type": "number"
          },
          "is_regular": {
            "type": "integer"
          },
          "floor_polygon": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "x": {
                  "type": "number"
                },
                "z": {
                  "type": "number"
                }
              },
              "required": ["x", "z"]
            }
          }
        }
      }
    }
  }
}