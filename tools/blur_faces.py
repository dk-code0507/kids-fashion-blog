"""
顔自動ぼかしツール - こどもふく図鑑
====================================
子どもの写真の顔を自動検出してぼかし処理します。

使い方:
  python tools/blur_faces.py 写真.jpg
  python tools/blur_faces.py 写真.jpg --output 出力.jpg
  python tools/blur_faces.py photos/  ← フォルダ内を一括処理

必要なもの（初回のみ）:
  pip install opencv-python-headless
"""

import sys
import os
import argparse
import urllib.request

try:
    import cv2
    import numpy as np
except ImportError:
    print("❌ opencv-python が必要です。以下を実行してください:")
    print("   pip install opencv-python-headless")
    sys.exit(1)


def download_cascade():
    """顔検出モデルをダウンロード（OpenCVに同梱）"""
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        print("❌ 顔検出モデルが見つかりません。")
        sys.exit(1)
    return cascade_path


def blur_faces_in_image(input_path: str, output_path: str, blur_level: int = 40) -> bool:
    """
    1枚の画像の顔をぼかす

    Args:
        input_path: 入力画像パス
        output_path: 出力画像パス
        blur_level: ぼかし強度（大きいほど強い、デフォルト40）
    Returns:
        成功したらTrue
    """
    img = cv2.imread(input_path)
    if img is None:
        print(f"  ❌ 画像を読み込めませんでした: {input_path}")
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 顔検出（正面）
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    faces_front = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
    )

    # 顔検出（横顔も）
    profile_cascade_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'
    profile_cascade = cv2.CascadeClassifier(profile_cascade_path)
    faces_profile = profile_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
    )

    # 全検出結果をまとめる
    all_faces = list(faces_front) + list(faces_profile)

    if len(all_faces) == 0:
        print(f"  ⚠️  顔が検出されませんでした（手動確認を推奨）: {os.path.basename(input_path)}")
    else:
        print(f"  ✅ {len(all_faces)} 個の顔を検出 → ぼかし処理")

    for (x, y, w, h) in all_faces:
        # 顔の周囲に余白を追加（頭全体をカバー）
        pad_x = int(w * 0.25)
        pad_y = int(h * 0.4)  # 上の余白を大きめに（髪の毛まで隠す）
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img.shape[1], x + w + pad_x)
        y2 = min(img.shape[0], y + h + int(h * 0.1))

        face_region = img[y1:y2, x1:x2]
        k = blur_level * 2 + 1  # カーネルサイズ（奇数必須）
        blurred = cv2.GaussianBlur(face_region, (k, k), blur_level)
        img[y1:y2, x1:x2] = blurred

    # 保存
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"  💾 保存: {output_path}")
    return True


def process_folder(folder_path: str, output_folder: str, blur_level: int):
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
        output_path = os.path.join(output_folder, f"{name}_blurred{ext}")
        print(f"🖼  {filename}")
        if blur_faces_in_image(input_path, output_path, blur_level):
            success += 1
        print()

    print(f"✅ 完了！ {success}/{len(files)} 枚処理しました")
    print(f"📂 出力先: {output_folder}")


def main():
    parser = argparse.ArgumentParser(
        description='子どもの写真の顔を自動でぼかすツール',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python tools/blur_faces.py photo.jpg
  python tools/blur_faces.py photo.jpg --output blurred.jpg
  python tools/blur_faces.py photos/        ← フォルダ一括処理
  python tools/blur_faces.py photo.jpg --blur 60   ← 強めのぼかし
        """
    )
    parser.add_argument('input', help='画像ファイルまたはフォルダのパス')
    parser.add_argument('--output', '-o', help='出力先（省略時は自動命名）')
    parser.add_argument('--blur', '-b', type=int, default=40, help='ぼかし強度 10〜80（デフォルト: 40）')

    args = parser.parse_args()

    print("=" * 50)
    print("  顔自動ぼかしツール - こどもふく図鑑")
    print("=" * 50)
    print()

    if os.path.isdir(args.input):
        # フォルダ処理
        output_dir = args.output or os.path.join(args.input, 'blurred')
        process_folder(args.input, output_dir, args.blur)
    elif os.path.isfile(args.input):
        # 1ファイル処理
        name, ext = os.path.splitext(args.input)
        output_path = args.output or f"{name}_blurred{ext}"
        print(f"🖼  {os.path.basename(args.input)}")
        blur_faces_in_image(args.input, output_path, args.blur)
    else:
        print(f"❌ ファイルまたはフォルダが見つかりません: {args.input}")
        sys.exit(1)


if __name__ == '__main__':
    main()
