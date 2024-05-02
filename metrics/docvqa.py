import requests

url = "https://rrc.cvc.uab.es/?com=downloads&action=download&ch=17&f=aHR0cHM6Ly9kYXRhc2V0cy5jdmMudWFiLmVzL3JyYy9Eb2NWUUEvVGFzazEvc3Bkb2N2cWFfaW1hZ2VzLnRhci5neg=="
response = requests.get(url, verify=False)

if response.status_code == 200:
    with open("output.zip", "wb") as file:
        file.write(response.content)
        print("Dataset downloaded successfully.")
else:
    print("Failed to download dataset.")
