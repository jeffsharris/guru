import re

# Open the input file and read its contents
with open("burbea-rss.txt", "r") as f:
    contents = f.read()

# Use a regular expression to find all URLs ending with ".mp3"
urls = re.findall(r'https?://[^\s]+\.mp3', contents)

# Open the output file and write the URLs to it
with open("talks.txt", "w") as f:
    for url in urls:
        f.write(url + "\n")