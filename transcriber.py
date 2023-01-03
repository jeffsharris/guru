import requests
import os
import whisper

# Set up the whisper model
model = whisper.load_model("small.en")


# Open the file and read its contents
with open("talks.txt", "r") as f:
    contents = f.read()

# Split the contents into a list of URLs
urls = contents.split("\n")

# Iterate through the list of URLs

for url in urls[4:]:
    response = requests.get(url)
    filename = url.split("/")[-1]
    with open(filename, "wb") as f:
        f.write(response.content)
    transcription = model.transcribe(filename)

    with open("transcriptions.txt", "a") as f:
        f.write(transcription["text"]+'\n\n')

    os.remove(filename)
    print(filename)