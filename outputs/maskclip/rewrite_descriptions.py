import json
import os

json_root_path = "/home/cigonzalez/code/open-ended_visual_recognition_benchmark/old_maskclip_descriptions"

for dataset in ["ade20k", "cityscapes"]:

    json_path = os.path.join(json_root_path, dataset, "descriptions.json")

    json_file = json.load(open(json_path))

    new_json = []

    for img_name in json_file:

        descriptions = json_file[img_name]

        img_name = img_name.split(".")[0]

        new_dict = {"image_id": img_name, "descriptions": descriptions}
        new_json.append(new_dict)


    json.dump(new_json, open(os.path.join("/home/cigonzalez/code/open-ended_visual_recognition_benchmark/outputs/maskclip", dataset, "descriptions.json"), "w"))