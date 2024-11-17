import os
import xml.etree.ElementTree as ET

xml_folder = r"C:\Users\ejekw\Documents\opt\python\objectDetectorSeg\src\images"
txt_folder = r"C:\Users\ejekw\Documents\opt\python\objectDetectorSeg\src\images\yollotxt"


# Ensure the output folder exists
os.makedirs(txt_folder, exist_ok=True)


def convert_to_yolo(size, box):
    """Convert Pascal VOC bounding box to YOLO format."""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    width = box[1] - box[0]
    height = box[3] - box[2]
    return (x_center * dw, y_center * dh, width * dw, height * dh)


# Define your class-to-ID mapping here
class_map = {"hard_disk_case": 0, "mifi": 1, "phone_charger": 2, 'smart_watch': 3, 'wireless_mouse':4}  # Add all your classes with their IDs

for xml_file in os.listdir(xml_folder):
    if not xml_file.endswith(".xml"):
        continue

    print(f"Processing file: {xml_file}")  # Debug line

    # Parse the XML file
    tree = ET.parse(os.path.join(xml_folder, xml_file))
    root = tree.getroot()

    # Get image size
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    # Prepare output file
    txt_filename = os.path.splitext(xml_file)[0] + ".txt"
    txt_filepath = os.path.join(txt_folder, txt_filename)

    print(f"Saving to: {txt_filepath}")  # Debug line

    with open(txt_filepath, "w") as txt_file:
        for obj in root.iter("object"):
            class_name = obj.find("name").text
            print(f"Found class: {class_name}")  # Debug line
            # Ensure the class exists in the class_map
            if class_name not in class_map:
                print(f"Class '{class_name}' not in class_map. Skipping.")
                continue

            class_id = class_map[class_name]

            # Get bounding box coordinates
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            xmax = float(bndbox.find("xmax").text)
            ymin = float(bndbox.find("ymin").text)
            ymax = float(bndbox.find("ymax").text)

            # Convert to YOLO format
            bbox = convert_to_yolo((width, height), (xmin, xmax, ymin, ymax))
            print(f"YOLO bbox: {bbox}")  # Debug line

            # Write to .txt file
            txt_file.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

print(f"XML folder: {xml_folder}")
print(f"TXT folder: {txt_folder}")

print("Conversion completed!")
