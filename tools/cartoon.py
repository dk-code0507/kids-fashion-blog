"""
カートゥーン・スケッチ変換ツール
==================================
写真をアニメ・漫画風・スケッチ風に変換します。
顔の特徴が自然に消えるのでプライバシー対策にもなります。

必要なもの（初回のみ）:
  pip install opencv-python-headless

使い方:
  python tools/cartoon.py 写真.jpg                  # カートゥーン（デフォルト）
  python tools/cartoon.py 写真.jpg --mode sketch    # 鉛筆スケッチ（白黒）
  python tools/cartoon.py 写真.jpg --mode soft      # やわらかイラスト風
  python tools/cartoon.py photos/                   # フォルダ一括処理
  python tools/cartoon.py 写真.jpg -o 出力.jpg      # 出力先指定
"""

import sys, os, argparse
try:
    import cv2
    import numpy as np
except ImportError:
    print("opencv が必要です: pip install opencv-python-headless")
    sys.exit(1)


# ─────────────────────────────────────────
#  変換モード
# ─────────────────────────────────────────

def cartoon(img):
    """
    カートゥーン化
    - 色を平坦化しつつエッジを保持（アニメ塗りっぽくなる）
    - 輪郭線を黒く強調
    """
    # 1. 色の平坦化（バイラテラルフィルタ・2回に抑えて高速化）
    smooth = img.copy()
    for _ in range(2):
        smooth = cv2.bilateralFilter(smooth, d=9, sigmaColor=75, sigmaSpace=75)

    # 2. エッジ検出
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(
        gray_blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=9, C=4
    )
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # 3. 合成（平坦化した色 × エッジマスク）
    result = cv2.bitwise_and(smooth, edges_bgr)
    return result


def soft_illust(img):
    """
    やわらかイラスト風
    - パステル調・水彩画っぽい仕上がり
    - エッジを残しつつ色をふんわりさせる
    """
    # 色の平坦化（強め）
    smooth = img.copy()
    for _ in range(3):
        smooth = cv2.bilateralFilter(smooth, d=15, sigmaColor=80, sigmaSpace=80)

    # 明るさを少し上げてパステル調に
    hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= 0.75   # 彩度を下げる（やわらかく）
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1 + 10, 0, 255)  # 明度を上げる
    smooth = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # ソフトなエッジ
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Laplacian(gray_blur, cv2.CV_8U, ksize=5)
    _, edge_mask = cv2.threshold(edges, 20, 255, cv2.THRESH_BINARY_INV)
    edge_mask_bgr = cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR)

    result = cv2.bitwise_and(smooth, edge_mask_bgr)

    # 全体にほんのり白を重ねてふんわり感を出す
    white = np.full_like(result, 255)
    result = cv2.addWeighted(result, 0.88, white, 0.12, 0)
    return result


def sketch(img):
    """
    鉛筆スケッチ風（白黒）
    - 輪郭線だけ残す
    - 顔の特徴がほぼ消えるので匿名性が高い
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ぼかしとの差分でエッジを抽出
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    diff = cv2.divide(gray, blur, scale=256.0)

    # コントラスト調整
    result_gray = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

    # BGR に戻す（カラーファイルとして保存するため）
    result = cv2.cvtColor(result_gray, cv2.COLOR_GRAY2BGR)
    return result


# ─────────────────────────────────────────
#  ファイル処理
# ─────────────────────────────────────────

MODES = {
    'cartoon': cartoon,
    'soft': soft_illust,
    'sketch': sketch,
}

def process(src, dst, mode, max_px=1600):
    img = cv2.imread(src)
    if img is None:
        print(f"  NG: 読み込めません → {src}")
        return False

    # ブログ用途には1600px以内で十分。大きい写真は縮小してから処理（高速化）
    h, w = img.shape[:2]
    if max(h, w) > max_px:
        scale = max_px / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        print(f"  縮小: {w}x{h} → {img.shape[1]}x{img.shape[0]}")

    fn = MODES.get(mode, cartoon)
    result = fn(img)

    os.makedirs(os.path.dirname(dst) if os.path.dirname(dst) else '.', exist_ok=True)
    cv2.imwrite(dst, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  -> {dst}")
    return True


def process_folder(src_dir, dst_dir, mode):
    exts = ('.jpg', '.jpeg', '.png', '.webp')
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(exts)]
    if not files:
        print(f"画像が見つかりません: {src_dir}")
        return
    os.makedirs(dst_dir, exist_ok=True)
    print(f"{len(files)} 枚処理します\n")
    ok = 0
    for f in files:
        name, ext = os.path.splitext(f)
        print(f"[{f}]")
        if process(os.path.join(src_dir, f),
                   os.path.join(dst_dir, f"{name}_{mode}{ext}"),
                   mode):
            ok += 1
    print(f"\n完了: {ok}/{len(files)} 枚 → {dst_dir}")


# ─────────────────────────────────────────
#  エントリポイント
# ─────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='写真をカートゥーン・スケッチ風に変換',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
モード:
  cartoon  アニメ・漫画風（デフォルト）
  soft     やわらかイラスト・水彩風
  sketch   鉛筆スケッチ（白黒・最も匿名性が高い）

例:
  py tools/cartoon.py 写真.jpg
  py tools/cartoon.py 写真.jpg --mode sketch
  py tools/cartoon.py 写真.jpg --mode soft
  py tools/cartoon.py photos/ --mode cartoon
        """
    )
    p.add_argument('input', help='画像ファイルまたはフォルダ')
    p.add_argument('--mode', '-m', choices=['cartoon', 'soft', 'sketch'],
                   default='cartoon', help='変換モード（デフォルト: cartoon）')
    p.add_argument('--output', '-o', help='出力先（省略時は自動命名）')
    args = p.parse_args()

    print("=" * 40)
    print(f"  カートゥーン変換 [{args.mode}]")
    print("=" * 40)
    print()

    if os.path.isdir(args.input):
        out = args.output or os.path.join(args.input, 'cartoon')
        process_folder(args.input, out, args.mode)
    elif os.path.isfile(args.input):
        name, ext = os.path.splitext(args.input)
        out = args.output or f"{name}_{args.mode}{ext}"
        process(args.input, out, args.mode, )
    else:
        print(f"見つかりません: {args.input}")
        sys.exit(1)


if __name__ == '__main__':
    main()
