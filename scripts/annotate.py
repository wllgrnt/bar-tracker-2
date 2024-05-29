import matplotlib.pyplot as plt
import requests

from PIL import Image
from transformers import pipeline

PROMPT = "a cat. a remote control"
# colors for visualization
COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933],
]


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


pipe = pipeline(task="zero-shot-object-detection", model="IDEA-Research/grounding-dino-tiny")

results = pipe(
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    candidate_labels=[PROMPT],
    threshold=0.3,
)

print(results)

image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)


scores, labels, boxes = [], [], []
for result in results:
    scores.append(result["score"])
    labels.append(result["label"])
    boxes.append(tuple(result["box"].values()))

plot_results(image, scores, labels, boxes)
