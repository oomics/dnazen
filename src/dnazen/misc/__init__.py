import hashlib


def hash_file_md5(file_path) -> str:
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


def check_hash_of_file_md5(file_path, target_md5: str) -> bool:
    return hash_file_md5(file_path) == target_md5
