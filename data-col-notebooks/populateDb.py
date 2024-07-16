import os
import requests
import json

# Grid regions worked on by different team members
regions = {
    # "NorthGUC": "Evan",
    # "CAISO": "Noah",
    # "ISONE": "Noah",
    # "NYISO": "Evan",
    "SERPT": "Noah",
}
# Read a geojson file with json.load()
data_dir = os.path.join(os.path.dirname(__file__), "..", "otherDatasets")

# api url
api_url = "http://127.0.0.1:8000/gis/bulk_update/update_markers/"
print(data_dir)

# crawl throught the data directory
for root, dirs, files in os.walk(data_dir):
    if os.path.basename(root) not in regions:
        print(os.path.basename(root))
        continue
    for file in files:
        if file.endswith(".geojson"):
            # read the file
            with open(os.path.join(root, file), "r") as f:
                data = json.load(f)

                # Print the base dir of the file
                parent_dir = os.path.basename(root)
                created_by = regions[parent_dir]
                updated_by = regions[parent_dir]

                # Construct the payload
                substation_id = data["features"][0]["properties"]["SS_ID"]
                print(data["features"][0]["properties"]["markers"])
                markers = []
                for i, marker in enumerate(
                    data["features"][0]["geometry"]["coordinates"]
                ):
                    try:
                        markers.append(
                            {
                                "label": data["features"][0]["properties"]["markers"][
                                    i
                                ]["type"],
                                "latitude": marker[1],
                                "longitude": marker[0],
                            }
                        )
                    except KeyError:
                        markers.append(
                            {
                                "label": data["features"][0]["properties"]["markers"][
                                    i
                                ]["label"],
                                "latitude": marker[1],
                                "longitude": marker[0],
                            }
                        )

                payload = {
                    "substation_id": substation_id,
                    "created_by": created_by,
                    "updated_by": updated_by,
                    "markers": markers,
                }

                # Print payload for debugging
                print(json.dumps(payload, indent=2))
                # Post the payload to the API
                response = requests.post(api_url, json=payload)
                print(response.status_code, response.json())
