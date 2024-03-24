import requests

search_url = "https://api.sketchfab.com/v3/search"


def search_results(params):
    r = requests.get(search_url, params)

    data = None
    try:
        data = r.json()
    except ValueError:
        pass

    assert r.ok, f"Search failed: {r.status_code} - {data}"

    return r.json()["results"]
