import os
import matplotlib.pyplot as plt
import torch


from PIL import Image
from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection


# PROMPT = "a cat."
PROMPT = "a round weight."
# colors for visualization
COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933],
]

IMAGE_PATH = "assets/snatch_photo.png"
# IMAGE_PATH = "http://images.cocodataset.org/val2017/000000039769.jpg"
assert os.path.exists(IMAGE_PATH)


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


if __name__ == "__main__":

    processor = GroundingDinoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")

    # image = Image.open(requests.get(IMAGE_PATH, stream=True).raw)
    image = Image.open(IMAGE_PATH).convert("RGB")
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
    plot_results(image, results["scores"], results["labels"], results["boxes"])
