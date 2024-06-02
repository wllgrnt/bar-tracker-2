import os
import cv2
import matplotlib.pyplot as plt
import torch
from PIL import Image
from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection

PROMPT = "a round weight."
COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933],
]

VIDEO_PATH = "assets/snatch.mp4"
assert os.path.exists(VIDEO_PATH)


def plot_results(pil_img, scores, labels, boxes):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax), c in zip(scores, labels, boxes, colors):
        ax.add_patch(
            plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3)
        )
        label = f"{PROMPT}: {score:0.2f}"
        ax.text(xmin, ymin, label, fontsize=15, bbox=dict(facecolor="yellow", alpha=0.5))
    plt.axis("off")
    plt.show()


def detect_plate_in_frame(frame, processor, model):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, text=PROMPT, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    width, height = image.size
    postprocessed_outputs = processor.post_process_grounded_object_detection(
        outputs,
        input_ids=inputs.input_ids,
        target_sizes=[(height, width)],
        box_threshold=0.3,
        text_threshold=0.1,
    )
    results = postprocessed_outputs[0]
    return results["boxes"]


def track_plate(video_path, initial_box):
    cap = cv2.VideoCapture(video_path)
    tracker = (
        cv2.TrackerCSRT_create()
    )  # this is what needs -contrib-, since TrackerMIL doesn't work.
    ret, frame = cap.read()

    if not ret:
        print("Failed to read video")
        return

    x_min, y_min, width, height = initial_box
    tracker.init(frame, (x_min, y_min, width, height))
    barpath = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        success, box = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in box]
            barpath.append((x + w / 2, y + h / 2))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (x + w // 2, y + h // 2), 5, (0, 0, 255), -1)

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return barpath


if __name__ == "__main__":
    processor = GroundingDinoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")

    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    cap.release()

    if ret:
        boxes = detect_plate_in_frame(frame, processor, model)
        if len(boxes) > 0:
            initial_box = boxes[0]
            x_min, y_min, x_max, y_max = [int(coord) for coord in initial_box]
            width, height = x_max - x_min, y_max - y_min
            initial_box = (x_min, y_min, width, height)
            barpath = track_plate(VIDEO_PATH, initial_box)

        else:
            print("No plate detected in the first frame.")
    else:
        print("Failed to read video.")
