import requests
import shutil
import os

from glb_to_obj import import_glb, export_obj
import auth

download_url = "https://api.sketchfab.com/v3/models/{}/download"


def _get_download_url(uid):
    print(f"Getting download url for uid {uid}")
    r = requests.get(
        download_url.format(uid),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Token {auth.__API_TOKEN}",
        },
    )

    try:
        data = r.json()
    except ValueError:
        pass

    assert r.ok, f"Failed to get url {uid}: {r.status_code} - {data}"

    assert "gltf" in data, f"'gltf' field not: {data}"
    gltf = data.get("gltf")

    assert "url" in gltf, f"'url' field not found: {data}"
    url = gltf.get("url")

    assert "size" in gltf, f"'size' field not found: {data}"
    size = gltf.get("size")

    return {"url": url, "size": size}


def download_model(model_uid, file_path, id):
    data = _get_download_url(model_uid)

    download_path = os.path.join(file_path, f"{model_uid}.zip")
    
    assert data['size'] / (1024 * 1024) < 100, "Too large model"
    
    print(f"Downloading model, size {(data['size'] / (1024 * 1024)):.1f}MB")
    with requests.get(data["url"], stream=True) as r:
        with open(download_path, "wb") as f:
            shutil.copyfileobj(r.raw, f)

    shutil.unpack_archive(download_path, file_path, "zip")
    os.unlink(download_path)

    target_location = os.path.join(file_path, "scene.gltf")
    return target_location


def convert_to_obj(input_path, id):
    print(input_path)
    import_glb(input_path)
    out_path = os.path.join(os.path.dirname(input_path), f"scene{id}.obj")
    export_obj(out_path)
    

def clear_extra_files(file_path):
    for file in os.listdir(file_path):
        current_file_path = os.path.join(file_path, file)
        if not current_file_path.endswith(".obj"):     
            if os.path.isfile(current_file_path):
                os.remove(current_file_path)
            if os.path.isdir(current_file_path):
                shutil.rmtree(current_file_path)
