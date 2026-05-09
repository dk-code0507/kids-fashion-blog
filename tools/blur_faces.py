"""
顔・背景ぼかしツール
====================================
子どもの写真の顔検出・背景ぼかし処理をします。

【初回セットアップ】
  pip install mediapipe opencv-python-headless

【使い方】
  python tools/blur_faces.py 写真.jpg                   # 顔だけぼかす
  python tools/blur_faces.py 写真.jpg --mode bg         # 背景だけぼかす
  python tools/blur_faces.py 写真.jpg --mode both       # 顔+背景 両方（おすすめ）
  python tools/blur_faces.py photos/  --mode both       # フォルダ一括
  python tools/blur_faces.py 写真.jpg --debug           # 検出範囲を確認
"""

import sys
import os
import argparse

try:
    import cv2
    import numpy as np
except ImportError:
    print("❌ opencv が必要です: pip install opencv-python-headless")
    sys.exit(1)

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False


# ============================================================
#  顔検出
# ============================================================

def detect_faces_mediapipe(img):
    """MediaPipe で顔検出（精度高・横顔・子どもも対応）"""
    mp_face = mp.solutions.face_detection
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = []
    # model_selection=0: 2m以内の近距離向け（子ども写真に最適）
    # model_selection=1: 遠距離向け（複数人・引きの写真）
    for model in [0, 1]:
        with mp_face.FaceDetection(model_selection=model, min_detection_confidence=0.3) as detector:
            result = detector.process(img_rgb)
            if result.detections:
                for det in result.detections:
                    bb = det.location_data.relative_bounding_box
                    x = max(0, int(bb.xmin * w))
                    y = max(0, int(bb.ymin * h))
                    fw = int(bb.width * w)
                    fh = int(bb.height * h)
                    faces.append((x, y, fw, fh))

    # 重複を除去（近い座標のものは1つにまとめる）
    return deduplicate_faces(faces)


def detect_faces_opencv(img):
    """フォールバック: OpenCV Haar（精度は低め）"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    h, w = img.shape[:2]
    min_face = max(20, min(h, w) // 12)

    all_faces = []
    for name in ['haarcascade_frontalface_default.xml', 'haarcascade_frontalface_alt.xml',
                 'haarcascade_frontalface_alt2.xml', 'haarcascade_profileface.xml']:
        path = cv2.data.haarcascades + name
        if not os.path.exists(path):
            continue
        cascade = cv2.CascadeClassifier(path)
        detected = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3,
                                            minSize=(min_face, min_face))
        if len(detected) > 0:
            all_faces.extend(detected.tolist())

    return deduplicate_faces(all_faces)


def deduplicate_faces(faces, iou_thresh=0.4):
    """重複するバウンディングボックスをまとめる"""
    if not faces:
        return []
    boxes = np.array(faces, dtype=float)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = areas.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        iw = np.maximum(0, ix2 - ix1)
        ih = np.maximum(0, iy2 - iy1)
        iou = (iw * ih) / (areas[i] + areas[order[1:]] - iw * ih + 1e-6)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return [boxes[i].astype(int).tolist() for i in keep]


# ============================================================
#  ぼかし処理
# ============================================================

def apply_face_blur(img, x, y, w, h, blur_level=55):
    """顔領域にぼかし＋モザイクを適用"""
    pad_x = int(w * 0.4)
    pad_y_top = int(h * 0.7)  # 上は多め（髪の毛まで隠す）
    pad_y_bot = int(h * 0.25)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y_top)
    x2 = min(img.shape[1], x + w + pad_x)
    y2 = min(img.shape[0], y + h + pad_y_bot)

    if x2 <= x1 or y2 <= y1:
        return

    region = img[y1:y2, x1:x2].copy()
    rh, rw = region.shape[:2]

    # ガウシアンぼかし
    k = blur_level * 2 + 1
    blurred = cv2.GaussianBlur(region, (k, k), blur_level)

    # モザイク（ピクセル化）
    mosaic_scale = max(1, min(rh, rw) // 8)
    small = cv2.resize(blurred, (max(1, rw // mosaic_scale), max(1, rh // mosaic_scale)),
                       interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)

    # ぼかし＋モザイク合成
    img[y1:y2, x1:x2] = cv2.addWeighted(blurred, 0.4, mosaic, 0.6, 0)


def apply_background_blur(img, blur_level=35):
    """MediaPipe Selfie Segmentation で背景だけぼかす"""
    if not HAS_MEDIAPIPE:
        print("  ⚠️  背景ぼかしには mediapipe が必要: pip install mediapipe")
        return img

    mp_seg = mp.solutions.selfie_segmentation
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_seg.SelfieSegmentation(model_selection=1) as seg:
        result = seg.process(img_rgb)
        mask = result.segmentation_mask  # 0.0〜1.0（人物=1.0）

    # マスクをスムーズに（境界をぼかす）
    mask_smooth = cv2.GaussianBlur(mask, (21, 21), 11)
    mask_3ch = np.stack([mask_smooth] * 3, axis=-1).astype(np.float32)

    # 背景をぼかす
    k = blur_level * 2 + 1
    bg_blurred = cv2.GaussianBlur(img, (k, k), blur_level)

    # 合成: 人物は元画像、背景はぼかし
    img_float = img.astype(np.float32)
    bg_float = bg_blurred.astype(np.float32)
    output = (img_float * mask_3ch + bg_float * (1.0 - mask_3ch)).astype(np.uint8)
    return output


# ============================================================
#  メイン処理
# ============================================================

def process_image(input_path: str, output_path: str,
                  mode: str = 'face', blur_level: int = 55,
                  bg_blur: int = 35, debug: bool = False) -> bool:
    """
    1枚の画像を処理する

    mode: 'face' | 'bg' | 'both'
    """
    img = cv2.imread(input_path)
    if img is None:
        print(f"  ❌ 画像を読み込めませんでした: {input_path}")
        return False

    # --- 背景ぼかし ---
    if mode in ('bg', 'both') and not debug:
        print(f"  🌆 背景をぼかし中...")
        img = apply_background_blur(img, bg_blur)

    # --- 顔検出・ぼかし ---
    if mode in ('face', 'both') or debug:
        if HAS_MEDIAPIPE:
            faces = detect_faces_mediapipe(img)
            detector_name = 'MediaPipe'
        else:
            faces = detect_faces_opencv(img)
            detector_name = 'OpenCV (精度低め)'

        if len(faces) == 0:
            print(f"  ⚠️  顔が検出されませんでした [{detector_name}]")
            print(f"      → --debug で確認、または写真を送ってください")
        else:
            print(f"  ✅ {len(faces)} 個の顔を検出 [{detector_name}]")

        for face in faces:
            x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            if debug:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                pad_x = int(w * 0.4)
                pad_y_top = int(h * 0.7)
                pad_y_bot = int(h * 0.25)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y_top)
                x2 = min(img.shape[1], x + w + pad_x)
                y2 = min(img.shape[0], y + h + pad_y_bot)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            else:
                apply_face_blur(img, x, y, w, h, blur_level)

    # --- 保存 ---
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  💾 保存: {output_path}")
    return True


def process_folder(folder_path: str, output_folder: str,
                   mode: str, blur_level: int, bg_blur: int, debug: bool):
    """フォルダ内の画像を一括処理"""
    extensions = ('.jpg', '.jpeg', '.png', '.webp')
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(extensions)]

    if not files:
        print(f"❌ 画像ファイルが見つかりません: {folder_path}")
        return

    print(f"📁 {len(files)} 枚の画像を処理します...\n")
    os.makedirs(output_folder, exist_ok=True)

    success = 0
    for filename in files:
        input_path = os.path.join(folder_path, filename)
        name, ext = os.path.splitext(filename)
        suffix = '_debug' if debug else f'_{mode}'
        output_path = os.path.join(output_folder, f"{name}{suffix}{ext}")
        print(f"🖼  {filename}")
        if process_image(input_path, output_path, mode, blur_level, bg_blur, debug):
            success += 1
        print()

    print(f"✅ 完了！ {success}/{len(files)} 枚処理しました")
    print(f"📂 出力先: {output_folder}")


def main():
    parser = argparse.ArgumentParser(
        description='子どもの写真の顔・背景をぼかすツール',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
モード:
  face  顔だけぼかす（デフォルト）
  bg    背景だけぼかす
  both  顔＋背景 両方ぼかす（プライバシー最強）

使用例:
  python tools/blur_faces.py photo.jpg
  python tools/blur_faces.py photo.jpg --mode both
  python tools/blur_faces.py photos/ --mode both
  python tools/blur_faces.py photo.jpg --debug
        """
    )
    parser.add_argument('input', help='画像ファイルまたはフォルダのパス')
    parser.add_argument('--output', '-o', help='出力先（省略時は自動命名）')
    parser.add_argument('--mode', '-m', choices=['face', 'bg', 'both'], default='face',
                        help='処理モード: face/bg/both（デフォルト: face）')
    parser.add_argument('--blur', '-b', type=int, default=55,
                        help='顔ぼかし強度 10〜80（デフォルト: 55）')
    parser.add_argument('--bg-blur', type=int, default=35,
                        help='背景ぼかし強度 10〜60（デフォルト: 35）')
    parser.add_argument('--debug', action='store_true',
                        help='検出範囲を枠で表示（ぼかしなし・確認用）')

    args = parser.parse_args()

    print("=" * 50)
    print("  顔・背景ぼかしツール")
    print("=" * 50)

    if not HAS_MEDIAPIPE:
        print()
        print("⚠️  mediapipe が見つかりません（精度が落ちます）")
        print("   pip install mediapipe  でインストールを推奨")

    print()

    if args.debug:
        print("🔍 デバッグモード: ぼかし処理なし、検出枠のみ表示")
        print()

    if os.path.isdir(args.input):
        out_dir = args.output or os.path.join(args.input, 'blurred')
        process_folder(args.input, out_dir, args.mode, args.blur, args.bg_blur, args.debug)
    elif os.path.isfile(args.input):
        name, ext = os.path.splitext(args.input)
        suffix = '_debug' if args.debug else f'_{args.mode}'
        output_path = args.output or f"{name}{suffix}{ext}"
        print(f"🖼  {os.path.basename(args.input)}")
        process_image(args.input, output_path, args.mode, args.blur, args.bg_blur, args.debug)
    else:
        print(f"❌ ファイルまたはフォルダが見つかりません: {args.input}")
        sys.exit(1)


if __name__ == '__main__':
    main()
