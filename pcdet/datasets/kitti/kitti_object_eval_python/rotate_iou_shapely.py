
import numpy as np
from shapely.geometry import Polygon
from shapely import affinity

def _box_to_poly(x, y, w, l, heading):
    # KITTI/PCDet 的 bev box 常用格式: [x, y, w, l, heading]
    # 这里以中心点 (x,y)，尺寸 (w,l)，绕 z 轴旋转 heading
    rect = Polygon([(-l/2, -w/2), (-l/2, w/2), (l/2, w/2), (l/2, -w/2)])
    rect = affinity.rotate(rect, heading * 180.0 / np.pi, origin=(0, 0), use_radians=False)
    rect = affinity.translate(rect, xoff=x, yoff=y)
    return rect

def rotate_iou_gpu_eval(boxes, qboxes, criterion=-1):
    """
    boxes:  (N, 5) [x, y, w, l, heading]
    qboxes: (K, 5)
    return: overlaps (N, K)
    criterion:
      -1: IoU
       0: intersection / area(boxes)
       1: intersection / area(qboxes)
       2: intersection
    """
    boxes = np.asarray(boxes)
    qboxes = np.asarray(qboxes)
    N = boxes.shape[0]
    K = qboxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float32)

    box_polys = [ _box_to_poly(b[0], b[1], b[2], b[3], b[4]) for b in boxes ]
    qbox_polys = [ _box_to_poly(b[0], b[1], b[2], b[3], b[4]) for b in qboxes ]

    box_areas = np.array([p.area for p in box_polys], dtype=np.float32)
    qbox_areas = np.array([p.area for p in qbox_polys], dtype=np.float32)

    for i in range(N):
        pi = box_polys[i]
        ai = box_areas[i]
        for j in range(K):
            pj = qbox_polys[j]
            inter = pi.intersection(pj).area
            if inter <= 0:
                continue
            if criterion == 2:
                overlaps[i, j] = inter
            elif criterion == 0:
                overlaps[i, j] = inter / max(ai, 1e-8)
            elif criterion == 1:
                overlaps[i, j] = inter / max(qbox_areas[j], 1e-8)
            else:
                union = ai + qbox_areas[j] - inter
                overlaps[i, j] = inter / max(union, 1e-8)
    return overlaps