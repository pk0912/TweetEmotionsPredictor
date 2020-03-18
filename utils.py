"""
Python file containing methods that are common requirement throughout the project
"""

from urllib import request


def download_and_write_to_file(url, filepath):
    response = request.urlopen(url)
    content = response.read()
    with open(filepath, "wb") as f:
        f.write(content)
