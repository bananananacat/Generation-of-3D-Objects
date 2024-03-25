from auth import set_api_token
from download import download_model, convert_to_obj, clear_extra_files
from search import search_results
import time

api_roken = "4adeb747fa4d4a3a890ea24a5640b262"
save_path = "./store"

names = [
    'kite',
    'lamp', 'map', 'notebook', 'orange', 'pen', 'quilt', 'rose', 'sock', 'table', 'umbrella',
    'vase', 'watermelon', 'xylophone', 'yogurt', 'zeppelin', 'apple', 'bird', 'cookie', 'duck', 'egg',
    'fork', 'glove', 'harp', 'island', 'jigsaw', 'kangaroo', 'lemon', 'mango', 'note', 'orange',
    'peach', 'quail', 'rocket', 'socks', 'turtle', 'unicorn', 'volcano', 'whale', 'xylophone', 'yak',
    'zebra', 'anchor', 'banana', 'coconut', 'dolphin', 'eagle', 'flowerpot', 'giraffe', 'hammer', 'igloo',
    'jellyfish', 'kite', 'lemonade', 'mushroom', 'notebook', 'owl', 'peacock', 'quilt', 'rainbow', 'squirrel',
    'tiger', 'umbrella', 'violin', 'watermelon', 'xylophone', 'yogurt', 'zeppelin', 'apple', 'book', 'cat',
    'desk', 'ear', 'fish', 'grape', 'hat', 'ink', 'jar', 'kite', 'lamp', 'map', 'notebook', 'orange', 'pen',
    'quilt', 'rose', 'sock', 'table', 'umbrella', 'vase', 'watermelon', 'xylophone', 'yogurt', 'zeppelin',
    'apple', 'bird', 'cookie', 'duck', 'egg', 'fork', 'glove', 'harp', 'island', 'jigsaw', 'kangaroo', 'lemon',
    'mango', 'note', 'orange', 'peach', 'quail', 'rocket', 'socks', 'turtle', 'unicorn', 'volcano', 'whale',
    'xylophone', 'yak', 'zebra', 'anchor', 'banana', 'coconut', 'dolphin', 'eagle', 'flowerpot', 'giraffe',
    'hammer', 'igloo', 'jellyfish', 'kite', 'lemonade', 'mushroom', 'notebook', 'owl', 'peacock', 'quilt',
    'rainbow', 'squirrel', 'tiger', 'umbrella', 'violin', 'watermelon', 'xylophone', 'yogurt', 'zeppelin'
]

counter = 288

for name in names:

    params = {
        "type": "models",
        "q": name, 
        "downloadable": True,
        "count": 24
    }

    models = search_results(params)
    if len(models) == 0:
        print("No models found")
        exit()

    set_api_token(api_roken)

    print (len(models))

    for i in range(len(models)):
        try:
            path = download_model(models[i]["uid"], save_path, counter)
            convert_to_obj(path, counter)
            clear_extra_files(save_path)
            time.sleep(2)
            counter += 1
        except Exception as e:
            print(e) 
