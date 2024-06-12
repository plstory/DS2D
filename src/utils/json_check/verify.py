from jsonschema import validate
from jsonschema.exceptions import ValidationError
from .schema import schema, strict_schema

def is_valid_json(json_data, strict=False):
    _schema = strict_schema if strict else schema
    try:
        validate(json_data, _schema)
        return True
    except ValidationError as e:
        return False