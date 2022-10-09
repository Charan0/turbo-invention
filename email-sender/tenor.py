import random
import requests

def parse_tenor(response: dict):
    if not response:
        return None
    results = response["results"]
    urls = []
    for result in results:
        available_media = result["media"]
        for media in available_media:
            if media["gif"]:
                urls.append(media["gif"]["url"])
    print("tenor request parsed successfully")
    return urls


def tenor_request(api_key: str, query_params: dict):
    try:
        r = requests.get(f"https://g.tenor.com/v1/search", params=query_params)
        if r.status_code == 200:
            response_data = r.json()
            print("tenor call successful with status code 200")
            return response_data
    except Exception as err:
        print(f"Encountered {err} when making request to tenor")
    return None


def get_gif(api_key: str, query_params: dict, index: int=None, url_only: bool=False):
    tenor_response = tenor_request(api_key, query_params)
    gifs_list = parse_tenor(tenor_response)
    limit = query_params.get("limit", 1)
    if gifs_list is None:
        return False
    if not index:
        index = random.randint(0, limit-1)
    url = gifs_list[index]
    
    if url_only:
        return url

    try:
        r = requests.get(url)
        gif_data = r.content
        with open("attachment.gif", "wb") as gif_file:
            gif_file.write(gif_data)
        print(f"written gif to file successfully")
        return True
    except Exception as err:
        print(f"Encountered {err} when makding request to gif url at {url}")
    return False
    
    