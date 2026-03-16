import cv2
from PIL import Image
from HP_model.Hydraplus import HP
from ultralytics import YOLO
import torch
from torchvision.transforms import v2 as T
import sys
import os

sys.path.append(os.path.abspath('HP_model'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load HP-Net model
hp_model = HP(num_classes=26)
hp_model.load_state_dict(torch.load("HP_best.pth", map_location=device))
hp_model.to(device)
hp_model.eval()

# HP-Net transforms
transform = T.Compose([
    T.PILToTensor(),
    T.Resize((299, 299), antialias=True),
    T.ToDtype(torch.float32),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")

yolo_model.to(device)

# Attribute names (replace with your actual 26 attributes)
attribute_names = [
    "backpack", "hat", "sunglasses", "long_sleeve", "short_sleeve", 
    "jacket", "coat", "pants", "skirt", "dress", 
    "boots", "sneakers", "shoes", "scarf", "gloves",
    "mask", "belt", "tie", "watch", "umbrella",
    "bag", "phone", "handbag", "helmet", "tshirt",
    "jeans"
]

# Open video
cap = cv2.VideoCapture("ch037_20260108101547_002.mp4")
if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # YOLOv8 person detection
    results = yolo_model(frame_rgb)[0]
    boxes = results.boxes.xyxy.cpu().numpy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        person_crop = frame_rgb[y1:y2, x1:x2]

        if person_crop.size == 0:
            continue

        # HP-Net preprocessing
        person_img = Image.fromarray(person_crop)
        person_tensor = transform(person_img).unsqueeze(0).to(device)

        # HP-Net inference
        with torch.no_grad():
            outputs = hp_model(person_tensor)
            outputs = torch.sigmoid(outputs).cpu().numpy()[0]

        attributes = (outputs > 0.5).astype(int)

        # Get names of detected attributes
        detected_attrs = [name for idx, name in enumerate(attribute_names) if attributes[idx] == 1]
        attr_text = ", ".join(detected_attrs) if detected_attrs else "None"
        print("Detected Attributes:", attr_text)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Show attribute names above the box
        cv2.putText(frame, attr_text, (x1, max(y1-10,0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Display the live video
    cv2.imshow("HP-Net Attributes", frame)

    # Stop if 'q' pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()