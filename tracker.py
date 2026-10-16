"""Lab 2: bám theo vật trên video bằng điểm đặc trưng.

Khung hình đầu tiên của video là ảnh cận của vật cần theo dõi.
Chạy:  python tracker.py video.avi                 -> ghi results/video.mp4
       python tracker.py video.avi "Mona Lisa"     -> đổi chữ trên khung
"""
import os
import sys
import time

import cv2
import numpy as np

RATIO = 0.75          # ngưỡng ratio test của Lowe
MIN_POINTS = 12       # ít điểm khớp đúng hơn số này thì coi như không thấy vật
LK = dict(winSize=(21, 21), maxLevel=3,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


class Tracker:
    def __init__(self, first_frame):
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        self.corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

        # Tìm điểm đặc trưng trên mẫu ở 3 tỉ lệ (100%, 50%, 25%),
        # để vẫn nhận ra vật khi nó ở xa và trông nhỏ đi
        pts, descs = [], []
        for s in (1, 0.5, 0.25):
            kp, des = self.sift.detectAndCompute(
                cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), None)
            pts += [(p.pt[0] / s, p.pt[1] / s) for p in kp]   # tọa độ quy về ảnh mẫu gốc
            descs.append(des)
        self.obj_pts, self.obj_des = np.float32(pts), np.vstack(descs)

        self.prev = None       # khung xám trước đó
        self.flow_obj = None   # điểm bám theo optical flow: tọa độ trên mẫu
        self.flow_cur = None   # và tọa độ của chúng trên khung trước

    def homography(self, src, dst):
        """Homography từ mẫu sang khung hình; RANSAC loại cặp điểm sai. Trả về (H, mask) hoặc (None, None)."""
        if len(src) < MIN_POINTS:
            return None, None
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None or mask.sum() < MIN_POINTS:
            return None, None
        box = cv2.perspectiveTransform(self.corners, H)
        if not cv2.isContourConvex(box.astype(np.int32)):   # khung bị méo thành hình lõm = sai
            return None, None
        return H, mask.ravel() == 1

    def detect(self, gray):
        """Cách 1: so điểm SIFT của khung hình với điểm của mẫu."""
        kp, des = self.sift.detectAndCompute(gray, None)
        if des is None or len(kp) < 2:
            return None
        # Mỗi điểm mẫu tìm 2 điểm giống nhất; giữ lại nếu điểm 1 tốt hơn hẳn điểm 2
        good = [m for m, n in (p for p in self.matcher.knnMatch(self.obj_des, des, k=2) if len(p) == 2)
                if m.distance < RATIO * n.distance]
        src = self.obj_pts[[m.queryIdx for m in good]]
        dst = np.float32([kp[m.trainIdx].pt for m in good])
        return self.homography(src, dst)[0]

    def follow(self, gray):
        """Cách 2 (dự phòng): dời các điểm từ khung trước sang khung này bằng optical flow."""
        if self.flow_cur is None:
            return None
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, self.flow_cur, None, **LK)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev, nxt, None, **LK)
        # Giữ điểm đi tới rồi đi ngược lại vẫn về đúng chỗ cũ (sai lệch < 1 pixel)
        ok = (st.ravel() == 1) & (st2.ravel() == 1) & \
             (np.linalg.norm((back - self.flow_cur).reshape(-1, 2), axis=1) < 1)
        H, inl = self.homography(self.flow_obj[ok], nxt[ok])
        if H is not None:
            keep = ok.copy()
            keep[ok] = inl
            self.flow_obj, self.flow_cur = self.flow_obj[keep], nxt[keep]
        return H

    def seed_flow(self, gray, H):
        """Sau khi SIFT tìm thấy vật: chọn các góc dễ bám bên trong khung để optical flow dùng tiếp."""
        mask = np.zeros_like(gray)
        cv2.fillConvexPoly(mask, cv2.perspectiveTransform(self.corners, H).astype(np.int32), 255)
        pts = cv2.goodFeaturesToTrack(gray, 300, 0.01, 5, mask=mask)
        if pts is None:
            self.flow_cur = None
            return
        self.flow_cur = pts.astype(np.float32)
        self.flow_obj = cv2.perspectiveTransform(self.flow_cur, np.linalg.inv(H)).reshape(-1, 2)

    def update(self, frame):
        """Xử lý một khung hình. Trả về (4 góc của vật hoặc None, cách đã tìm ra)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H, method = self.detect(gray), "SIFT"
        if H is not None:
            self.seed_flow(gray, H)
        else:
            H, method = self.follow(gray), "flow"
            if H is None:
                self.flow_cur, method = None, None
        self.prev = gray
        return (cv2.perspectiveTransform(self.corners, H) if H is not None else None), method


def draw(frame, box, label):
    """Vẽ khung 4 góc và chữ chú thích dưới khung."""
    pts = box.astype(np.int32)
    cv2.polylines(frame, [pts], True, (0, 255, 0), 3)
    x, top, bottom = int(pts[:, 0, 0].min()), int(pts[:, 0, 1].min()), int(pts[:, 0, 1].max())
    # Chữ đặt dưới khung; nếu sát mép dưới video thì đặt lên trên khung
    org = (max(x, 5), bottom + 30 if bottom + 30 < frame.shape[0] else max(top - 10, 25))
    cv2.putText(frame, label, org, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
    cv2.putText(frame, label, org, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


def main(path, label):
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    if not ok:
        sys.exit(f"Cannot open video: {path}")
    tracker = Tracker(frame)

    os.makedirs("results", exist_ok=True)
    name = os.path.splitext(os.path.basename(path))[0]
    h, w = frame.shape[:2]
    out = cv2.VideoWriter(f"results/{name}.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                          cap.get(cv2.CAP_PROP_FPS) or 30, (w, h))

    count = {"SIFT": 0, "flow": 0, None: 0}
    start = time.perf_counter()
    while ok:
        box, method = tracker.update(frame)
        count[method] += 1
        if box is not None:
            draw(frame, box, label)
        out.write(frame)
        ok, frame = cap.read()
    elapsed = time.perf_counter() - start
    cap.release()
    out.release()

    total = sum(count.values())
    found = total - count[None]
    print(f"{name}: frames {total}, found {found} ({100 * found / total:.1f}%) "
          f"[SIFT {count['SIFT']}, flow {count['flow']}], speed {total / elapsed:.1f} fps")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python tracker.py video.avi [label]")
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "Object")
