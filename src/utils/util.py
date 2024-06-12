def natural_sort_key(s):
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

def list_folders(directory, use_natural_sort=True):
    import os
    folders = [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]
    if use_natural_sort:
        folders.sort(key=natural_sort_key)
    else:
        folders.sort(key=str.lower)
    return folders

def list_json_files(directory):
    import os
    json_files = [file for file in os.listdir(directory) if file.endswith('.json') and os.path.isfile(os.path.join(directory, file))]
    json_files.sort(key=natural_sort_key)
    return json_files