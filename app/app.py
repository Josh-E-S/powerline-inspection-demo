"""Gradio demo for the powerline inspection pipeline.

An uploaded image runs through the detector, then the condition
classifier on each crop of a defect-prone asset. A confidence threshold
splits the results into findings the model is confident about and ones
routed to a human, and either group can be exported as GeoJSON.

Inference happens once per image and the detections are cached, so moving
the threshold slider only re-filters results that already exist. Serving
uses ONNX Runtime alone, with no torch or Ultralytics dependency, which
keeps the deployed image small and the cold start short.
"""

import ast
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
IMGSZ = 640
CONF_FLOOR = 0.05
IOU_NMS = 0.45

# Detector class name -> fault-classifier class prefix (the five
# defect-prone asset types; all others are detect-only).
ASSET_MAP = {
    "glass insulator": "glass-insulator",
    "lightning rod suspension": "lightning-rod-suspension",
    "polymer insulator upper shackle": "polymer-insulator-upper-shackle",
    "vari-grip": "vari-grip",
    "yoke suspension": "yoke-suspension",
}

# Simulated corridor: approximate alignment of a transmission corridor
# east of Tucson, AZ. Coordinates are SIMULATED for demo purposes.
CORRIDOR = [(32.206, -110.745), (32.130, -110.601), (32.045, -110.451)]

# ---------------------------------------------------------------- models
import onnxruntime as ort  # noqa: E402

DET = ort.InferenceSession(
    str(HERE / "models" / "detector_int8.onnx"),
    providers=["CPUExecutionProvider"],
)
CLS = ort.InferenceSession(
    str(HERE / "models" / "classifier.onnx"),
    providers=["CPUExecutionProvider"],
)
CLS_CLASSES = json.loads(
    (HERE / "models" / "classifier_classes.json").read_text()
)
DET_INPUT = DET.get_inputs()[0].name
NAMES = ast.literal_eval(DET.get_modelmeta().custom_metadata_map["names"])
PR_TABLE = json.loads((HERE / "pr_table.json").read_text())

MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)


# ------------------------------------------------------------- inference
def letterbox(img: Image.Image):
    """Fit the image into a square canvas, matching training preprocessing.

    Returns the tensor plus the scale and padding needed to map boxes
    back onto the original image.
    """
    r = min(IMGSZ / img.width, IMGSZ / img.height)
    nw, nh = round(img.width * r), round(img.height * r)
    px, py = (IMGSZ - nw) // 2, (IMGSZ - nh) // 2
    canvas = Image.new("RGB", (IMGSZ, IMGSZ), (114, 114, 114))
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (px, py))
    arr = np.asarray(canvas, np.float32).transpose(2, 0, 1)[None] / 255.0
    return arr, r, px, py


def nms(boxes, scores, iou_thr):
    """Suppress overlapping boxes, keeping the highest-scoring ones."""
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_o = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (
            boxes[order[1:], 3] - boxes[order[1:], 1]
        )
        iou = inter / (area_i + area_o - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def classify_crop(img: Image.Image, box, prefix):
    """Classify the condition of one detected component.

    The classifier has a single head over every asset's conditions, so
    the logits are masked down to the ones valid for this asset before
    the softmax. A glass insulator cannot be diagnosed with a bird's
    nest.
    """
    crop = img.crop(box)
    r = 256 / min(crop.size)
    crop = crop.resize(
        (max(1, round(crop.width * r)), max(1, round(crop.height * r))),
        Image.BILINEAR,
    )
    left, top = (crop.width - 224) // 2, (crop.height - 224) // 2
    crop = crop.crop((left, top, left + 224, top + 224))
    x = np.asarray(crop, np.float32).transpose(2, 0, 1) / 255.0
    x = ((x - MEAN) / STD)[None]
    logits = CLS.run(None, {CLS.get_inputs()[0].name: x})[0][0]
    mask = np.array([c.startswith(prefix + "__") for c in CLS_CLASSES])
    logits = np.where(mask, logits, -1e9)
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    i = int(probs.argmax())
    return CLS_CLASSES[i].split("__", 1)[1], float(probs[i])


def run_pipeline(image: Image.Image):
    """Full inference. Returns a list of detection dicts, conf >= 0.05."""
    img = image.convert("RGB")
    x, r, px, py = letterbox(img)
    out = DET.run(None, {DET_INPUT: x})[0][0]  # (4+nc, anchors)
    boxes_xywh, scores = out[:4].T, out[4:].T  # (n,4), (n,nc)
    cls_ids = scores.argmax(1)
    confs = scores.max(1)
    m = confs >= CONF_FLOOR
    boxes_xywh, cls_ids, confs = boxes_xywh[m], cls_ids[m], confs[m]

    xy = np.empty_like(boxes_xywh)
    xy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    xy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    xy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    xy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    # Offsetting boxes by class id keeps NMS from suppressing an
    # overlapping detection of a genuinely different component.
    offset = cls_ids[:, None] * (IMGSZ * 2)
    keep = nms(xy + offset, confs, IOU_NMS)

    dets = []
    for i in keep:
        x1 = (xy[i, 0] - px) / r
        y1 = (xy[i, 1] - py) / r
        x2 = (xy[i, 2] - px) / r
        y2 = (xy[i, 3] - py) / r
        box = (
            max(0, int(x1)),
            max(0, int(y1)),
            min(img.width, int(x2)),
            min(img.height, int(y2)),
        )
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
            continue
        name = NAMES[int(cls_ids[i])]
        det = {
            "asset": name,
            "conf": round(float(confs[i]), 3),
            "box": box,
            "condition": None,
            "cond_conf": None,
        }
        if name in ASSET_MAP:
            cond, p = classify_crop(img, box, ASSET_MAP[name])
            det["condition"], det["cond_conf"] = cond, round(p, 3)
        dets.append(det)
    dets.sort(key=lambda d: -d["conf"])
    return dets


# -------------------------------------------------------------------- ui
def render(image, dets, thr):
    """Draw the detections and split them across the threshold.

    Called on every slider move, so it works from cached detections and
    never re-runs the models.
    """
    if image is None or dets is None:
        return None, [], [], pr_text(thr)
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    w = max(2, img.width // 400)
    accepted, queue = [], []
    for d in dets:
        ok = d["conf"] >= thr
        color = (0, 200, 83) if ok else (255, 145, 0)
        draw.rectangle(d["box"], outline=color, width=w)
        label = d["asset"] + (f" | {d['condition']}" if d["condition"] else "")
        cond = (
            f"{d['condition']} ({d['cond_conf']:.2f})"
            if d["condition"]
            else "n/a"
        )
        if ok:
            accepted.append([d["asset"], f"{d['conf']:.2f}", cond])
        else:
            crop = image.convert("RGB").crop(d["box"])
            crop.thumbnail((200, 200))
            queue.append(
                (np.asarray(crop), f"{label} | det conf {d['conf']:.2f}")
            )
    return img, accepted, queue, pr_text(thr)


def pr_text(thr):
    """Look up precision and recall for the current threshold."""
    p = float(np.interp(thr, PR_TABLE["thresholds"], PR_TABLE["precision"]))
    r = float(np.interp(thr, PR_TABLE["thresholds"], PR_TABLE["recall"]))
    return (
        f"**At threshold {thr:.2f}: precision {p:.3f}, recall {r:.3f}**  \n"
        "Measured on the held-out test set (2,626 images), "
        "macro-averaged over 17 classes. Dataset-level, not per-image."
    )


def analyze(image, thr):
    if image is None:
        return None, None, [], [], pr_text(thr)
    dets = run_pipeline(image)
    img, accepted, queue, pr = render(image, dets, thr)
    return dets, img, accepted, queue, pr


def reslider(image, dets, thr):
    return render(image, dets, thr)


def export_geojson(image, dets, thr):
    """Build a GeoJSON FeatureCollection from the current findings.

    Coordinates are simulated along a real corridor and labelled as such
    in every feature. The scatter is seeded from the detections so the
    same image always exports to the same place.
    """
    if not dets:
        return None
    seed = int(
        hashlib.sha256(
            (str(len(dets)) + str(dets[0]["box"])).encode()
        ).hexdigest()[:8],
        16,
    )
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 1)
    features = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for k, d in enumerate(dets):
        u = min(0.999, t + k * 0.004)
        seg = 0 if u < 0.5 else 1
        f = (u * 2) % 1.0
        (la1, lo1), (la2, lo2) = CORRIDOR[seg], CORRIDOR[seg + 1]
        lat = la1 + (la2 - la1) * f + float(rng.normal(0, 0.0004))
        lon = lo1 + (lo2 - lo1) * f + float(rng.normal(0, 0.0004))
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [round(lon, 6), round(lat, 6)],
                },
                "properties": {
                    "asset_class": d["asset"],
                    "condition": d["condition"] or "not assessed",
                    "detection_confidence": d["conf"],
                    "condition_confidence": d["cond_conf"],
                    "review_status": (
                        "auto_accepted"
                        if d["conf"] >= thr
                        else "routed_to_human_review"
                    ),
                    "timestamp": now,
                    "note": "coordinates SIMULATED along a transmission "
                    "corridor east of Tucson, AZ for demo purposes",
                },
            }
        )
    geo = {"type": "FeatureCollection", "features": features}
    path = Path(tempfile.mkdtemp()) / "findings.geojson"
    path.write_text(json.dumps(geo, indent=2))
    return str(path)


with gr.Blocks(title="Powerline Inspection Demo") as demo:
    gr.Markdown(
        "# Powerline inspection demo\n"
        "UAV image → component detection (YOLO11-s, INT8) → condition "
        "classification (EfficientNetV2-S) → threshold gate → human "
        "review queue → GeoJSON export. Drag the threshold to trade "
        "missed defects against reviewer workload. "
        "[Code and measured results](https://github.com/Josh-E-S/powerline-inspection-demo)"
    )
    dets_state = gr.State()
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="pil", label="UAV image")
            gr.Examples(
                [str(p) for p in sorted((HERE / "examples").glob("*.jpg"))],
                inputs=inp,
            )
            btn = gr.Button("Analyze", variant="primary")
            thr = gr.Slider(
                0.05,
                0.95,
                value=0.5,
                step=0.05,
                label="Confidence threshold (detection)",
            )
            pr_md = gr.Markdown(pr_text(0.5))
        with gr.Column(scale=2):
            out_img = gr.Image(
                label="Detections (green = auto-accepted, "
                "orange = routed to review)"
            )
            accepted_df = gr.Dataframe(
                headers=["asset", "det conf", "condition"],
                label="Auto-accepted findings",
                interactive=False,
            )
    gr.Markdown(
        "## Human review queue\nDetections below the threshold are "
        "routed here instead of being discarded."
    )
    queue_gal = gr.Gallery(
        label="Routed to human review", columns=6, height=240
    )
    with gr.Row():
        export_btn = gr.Button("Export findings (GeoJSON)")
        geo_file = gr.File(label="GeoJSON (simulated coordinates)")
    gr.Markdown(
        "Demo-scale replica; condition labels are image-level, coordinates "
        "are simulated, and the P/R readout is dataset-level. Dataset: "
        "[InsPLAD](https://github.com/andreluizbvs/InsPLAD) (CC BY-NC 3.0)."
    )

    btn.click(
        analyze,
        [inp, thr],
        [dets_state, out_img, accepted_df, queue_gal, pr_md],
    )
    thr.release(
        reslider,
        [inp, dets_state, thr],
        [out_img, accepted_df, queue_gal, pr_md],
    )
    export_btn.click(export_geojson, [inp, dets_state, thr], geo_file)

if __name__ == "__main__":
    demo.launch()
