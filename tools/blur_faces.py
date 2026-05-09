"""
顔スタンプ・ぼかしツール
================================================
必要なもの（初回のみ）:
  pip install opencv-python-headless

【使い方】

■ 手動で顔位置を指定してスタンプ（一番確実）:
  python tools/blur_faces.py 写真.jpg --stamp 25 18
  ↑ 「画像の左から25%、上から18%の位置」にスタンプ
  ※ 写真を見てだいたいの位置をパーセントで指定

■ 複数の顔に対応:
  python tools/blur_faces.py 写真.jpg --stamp 25 18 --stamp 60 20

■ スタンプの種類を変える:
  python tools/blur_faces.py 写真.jpg --stamp 25 18 --style star   # ⭐（デフォルト）
  python tools/blur_faces.py 写真.jpg --stamp 25 18 --style circle # 丸ぼかし
  python tools/blur_faces.py 写真.jpg --stamp 25 18 --style mosaic # モザイク

■ スタンプのサイズを変える（デフォルト18%）:
  python tools/blur_faces.py 写真.jpg --stamp 25 18 --size 22

■ 自動検出を試す（横顔は検出できないことがあります）:
  python tools/blur_faces.py 写真.jpg --auto
  python tools/blur_faces.py 写真.jpg --auto --style mosaic

■ フォルダ一括（自動検出）:
  python tools/blur_faces.py photos/ --auto
"""

import sys
import os
import argparse
import math

try:
    import cv2
    import numpy as np
except ImportError:
    print("opencv が必要です: pip install opencv-python-headless")
    sys.exit(1)


# ─────────────────────────────────────────
#  スタンプ描画
# ─────────────────────────────────────────

def draw_star(img, cx, cy, r):
    """黄色い星スタンプ"""
    pts = []
    for i in range(10):
        angle = math.radians(i * 36 - 90)
        radius = r if i % 2 == 0 else r * 0.45
        pts.append((int(cx + radius * math.cos(angle)),
                    int(cy + radius * math.sin(angle))))
    pts = np.array(pts, np.int32)
    # 影
    shadow = pts + np.array([4, 4])
    cv2.fillPoly(img, [shadow], (80, 80, 0))
    # 星本体
    cv2.fillPoly(img, [pts], (0, 210, 255))    # 黄色（BGR）
    cv2.polylines(img, [pts], True, (0, 160, 200), max(2, r // 20))


def draw_circle_blur(img, cx, cy, r):
    """丸ぼかし（グラデーション境界）"""
    H, W = img.shape[:2]
    mask = np.zeros((H, W), dtype=np.float32)
    cv2.circle(mask, (cx, cy), r, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (r | 1, r | 1), r // 3)
    mask3 = np.stack([mask] * 3, axis=-1)
    k = 61
    blurred = cv2.GaussianBlur(img, (k, k), 30)
    result = (img.astype(np.float32) * (1 - mask3) +
              blurred.astype(np.float32) * mask3).astype(np.uint8)
    img[:] = result


def draw_mosaic(img, cx, cy, r):
    """モザイク"""
    H, W = img.shape[:2]
    x1, y1 = max(0, cx - r), max(0, cy - r)
    x2, y2 = min(W, cx + r), min(H, cy + r)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]
    block = max(8, min(rh, rw) // 8)
    small = cv2.resize(roi, (max(1, rw // block), max(1, rh // block)),
                       interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    # 円形マスクで切り抜き
    circle_mask = np.zeros((rh, rw), dtype=np.float32)
    cv2.circle(circle_mask, (rw // 2, rh // 2), min(rh, rw) // 2, 1.0, -1)
    circle_mask = cv2.GaussianBlur(circle_mask, (21, 21), 10)
    m3 = np.stack([circle_mask] * 3, axis=-1)
    img[y1:y2, x1:x2] = (roi.astype(np.float32) * (1 - m3) +
                           mosaic.astype(np.float32) * m3).astype(np.uint8)


def apply_stamp(img, cx, cy, size_px, style):
    r = size_px // 2
    if style == 'star':
        draw_star(img, cx, cy, r)
    elif style == 'circle':
        draw_circle_blur(img, cx, cy, r)
    elif style == 'mosaic':
        draw_mosaic(img, cx, cy, r)


# ─────────────────────────────────────────
#  自動顔検出（横顔は苦手）
# ─────────────────────────────────────────

def detect_faces(img):
    H, W = img.shape[:2]
    MAX = 800
    sc = min(1.0, MAX / max(H, W))
    small = cv2.resize(img, (int(W * sc), int(H * sc))) if sc < 1.0 else img
    gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    sh, sw = small.shape[:2]
    min_sz = max(15, min(sh, sw) // 14)

    boxes = []
    for name in ['haarcascade_frontalface_default.xml',
                 'haarcascade_frontalface_alt.xml',
                 'haarcascade_frontalface_alt2.xml',
                 'haarcascade_profileface.xml']:
        path = cv2.data.haarcascades + name
        if not os.path.exists(path):
            continue
        hits = cv2.CascadeClassifier(path).detectMultiScale(
            gray, 1.05, 2, minSize=(min_sz, min_sz))
        if len(hits) > 0:
            for (x, y, w, h) in hits:
                boxes.append((int(x/sc), int(y/sc), int(w/sc), int(h/sc)))

    return nms(boxes)


def nms(boxes, thresh=0.4):
    if not boxes:
        return []
    a = np.array(boxes, dtype=float)
    x1,y1,x2,y2 = a[:,0], a[:,1], a[:,0]+a[:,2], a[:,1]+a[:,3]
    areas = (x2-x1)*(y2-y1)
    order = areas.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]; keep.append(i)
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        iou = (np.maximum(0,ix2-ix1)*np.maximum(0,iy2-iy1)) / \
              (areas[i]+areas[order[1:]]-np.maximum(0,ix2-ix1)*np.maximum(0,iy2-iy1)+1e-6)
        order = order[np.where(iou<=thresh)[0]+1]
    return [tuple(a[i].astype(int)) for i in keep]


# ─────────────────────────────────────────
#  メイン処理
# ─────────────────────────────────────────

def process(src, dst, stamps, size_pct, style, auto):
    img = cv2.imread(src)
    if img is None:
        print(f"  NG: {src}")
        return False
    H, W = img.shape[:2]

    positions = []  # (cx, cy, size_px)

    # 手動スタンプ
    for (px_pct, py_pct) in stamps:
        cx = int(W * px_pct / 100)
        cy = int(H * py_pct / 100)
        size_px = int(min(W, H) * size_pct / 100)
        positions.append((cx, cy, size_px))

    # 自動検出
    if auto:
        faces = detect_faces(img)
        if faces:
            print(f"  OK: {len(faces)} 個検出")
            for (x, y, w, h) in faces:
                cx = x + w // 2
                cy = y + h // 2
                size_px = int(max(w, h) * 1.6)
                positions.append((cx, cy, size_px))
        else:
            print(f"  --: 顔が検出されませんでした")
            print(f"      --stamp X Y で手動指定してください")

    if not positions:
        print(f"  ?? スタンプ位置が指定されていません")
        print(f"     例: --stamp 25 18  (左から25%, 上から18%)")
        return False

    for (cx, cy, size_px) in positions:
        apply_stamp(img, cx, cy, size_px, style)
        print(f"  スタンプ: ({cx}, {cy}) サイズ={size_px}px スタイル={style}")

    os.makedirs(os.path.dirname(dst) if os.path.dirname(dst) else '.', exist_ok=True)
    cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  -> {dst}")
    return True


def process_folder(src_dir, dst_dir, stamps, size_pct, style, auto):
    exts = ('.jpg', '.jpeg', '.png', '.webp')
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(exts)]
    if not files:
        print(f"画像が見つかりません: {src_dir}")
        return
    os.makedirs(dst_dir, exist_ok=True)
    ok = 0
    for f in files:
        name, ext = os.path.splitext(f)
        print(f"\n[{f}]")
        if process(os.path.join(src_dir, f),
                   os.path.join(dst_dir, f"{name}_stamp{ext}"),
                   stamps, size_pct, style, auto):
            ok += 1
    print(f"\n完了: {ok}/{len(files)} 枚 → {dst_dir}")


def main():
    p = argparse.ArgumentParser(
        description='顔スタンプ・ぼかしツール',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  py tools/blur_faces.py 写真.jpg --stamp 25 18
  py tools/blur_faces.py 写真.jpg --stamp 25 18 --style mosaic
  py tools/blur_faces.py 写真.jpg --stamp 25 18 --stamp 60 20
  py tools/blur_faces.py 写真.jpg --auto
  py tools/blur_faces.py photos/ --auto --style mosaic
        """
    )
    p.add_argument('input')
    p.add_argument('--output', '-o')
    p.add_argument('--stamp', nargs=2, type=float, metavar=('X%', 'Y%'),
                   action='append', default=[],
                   help='スタンプ位置を左から%%、上から%%で指定（複数指定可）')
    p.add_argument('--size', type=float, default=18,
                   help='スタンプサイズ（画像短辺に対する%%、デフォルト18）')
    p.add_argument('--style', choices=['star', 'circle', 'mosaic'], default='star',
                   help='スタンプ種類: star/circle/mosaic（デフォルト: star）')
    p.add_argument('--auto', action='store_true',
                   help='自動顔検出を使う（横顔は検出できないことがあります）')
    args = p.parse_args()

    print("=" * 40)
    print("  顔スタンプ・ぼかしツール")
    print("=" * 40)
    print()

    if not args.stamp and not args.auto:
        print("使い方:")
        print("  手動: py tools/blur_faces.py 写真.jpg --stamp 25 18")
        print("         ↑ 画像の左から25%、上から18%の位置にスタンプ")
        print("  自動: py tools/blur_faces.py 写真.jpg --auto")
        print()
        p.print_help()
        return

    if os.path.isdir(args.input):
        out = args.output or os.path.join(args.input, 'stamped')
        process_folder(args.input, out, args.stamp, args.size, args.style, args.auto)
    elif os.path.isfile(args.input):
        name, ext = os.path.splitext(args.input)
        out = args.output or f"{name}_stamp{ext}"
        process(args.input, out, args.stamp, args.size, args.style, args.auto)
    else:
        print(f"見つかりません: {args.input}")
        sys.exit(1)


if __name__ == '__main__':
    main()
