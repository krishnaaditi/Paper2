# eval_chair.py
import json
import re
import argparse
from pycocotools.coco import COCO


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_json", required=True)
    ap.add_argument("--instances_json", required=True)
    args = ap.parse_args()

    preds = json.load(open(args.pred_json))
    coco = COCO(args.instances_json)

    cats = coco.loadCats(coco.getCatIds())
    id_to_name = {c["id"]: c["name"].lower() for c in cats}

    img_to_gt = {}
    for img_id in coco.getImgIds():
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
        img_to_gt[img_id] = set(id_to_name[a["category_id"]] for a in anns)

    sent_hall = 0
    inst_hall = 0
    total_sent = 0
    total_pred = 0

    for row in preds:
        img_id = row["image_id"]
        gt = img_to_gt[img_id]

        tokens = set(tokenize(row["caption"]))

        pred_objs = []
        for obj in id_to_name.values():
            if obj in tokens:
                pred_objs.append(obj)

        total_sent += 1
        hall_flag = False

        for obj in pred_objs:
            total_pred += 1
            if obj not in gt:
                inst_hall += 1
                hall_flag = True

        if hall_flag:
            sent_hall += 1

    chair_s = 100 * sent_hall / max(total_sent, 1)
    chair_i = 100 * inst_hall / max(total_pred, 1)

    print({
        "CHAIR_s": round(chair_s, 4),
        "CHAIR_i": round(chair_i, 4)
    })


if __name__ == "__main__":
    main()
