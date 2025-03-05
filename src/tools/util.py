import os


def find_file(root_folder, target_filename):
    for dirpath, dirnames, filenames in os.walk(root_folder):
        if target_filename in filenames:
            yield os.path.join(dirpath, target_filename)
