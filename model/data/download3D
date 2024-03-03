import os
import requests

api_key = "2d5aab9ed87f4c7cb2257a3e05a243ef"

download_dir = "downloaded_models_OBJ"
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

def download_models():
    api_url = "https://api.sketchfab.com/v3/models"

    params = {
        "key": api_key,
        "per_page": 100,
        "page": 1
    }

    downloaded_count = 0
    max_models = 4000

    while downloaded_count < max_models:
        response = requests.get(api_url, params=params)
        if response.status_code == 200:
            models = response.json()["results"]
            if not models:
                break

            for model in models:
                model_id = model["uid"]
                try:
                    download_link = model["download"]["obj"]["url"]
                except KeyError:
                    continue

                file_name = f"{model_id}.obj"
                file_path = os.path.join(download_dir, file_name)

                response = requests.get(download_link)
                if response.status_code == 200:
                    with open(file_path, "wb") as f:
                        f.write(response.content)
                    downloaded_count += 1
                else:
                    print(f"Ошибка при загрузке модели {model_id}.")

        params["page"] += 1

    print(f"Всего скачано {downloaded_count} моделей.")

download_models()
