# pt_FaceVisualize

Web カメラや読み込んだ画像・動画から顔情報を推定し、表情や顔のブレンドシェイプを可視化する Web アプリです。

## Demo

GitHub Pages で公開している場合は、以下からアクセスできます。

<https://sooochan0809.github.io/pt_FaceVisualize/>

## Features

- **Emotion**
  - `face-api.js` を使って顔の表情を推定
  - 推定結果をバー、レーダーチャート、感情円環モデル、色フィールドで可視化
  - Web カメラ、画像ファイル、動画ファイルに対応
- **BlendShape**
  - MediaPipe Face Landmarker を使って顔ランドマークとブレンドシェイプを推定
  - 顔ランドマークをキャンバスに描画
  - 各ブレンドシェイプのスコアをリアルタイム表示
  - 無表情を基準にキャリブレーションし、基準からの変化値を表示

## Directory Structure

```text
.
├── index.html
├── emotion/
│   └── index.html
├── blendshape/
│   └── index.html
└── faceAPI/
    ├── face-api.min.js
    └── models/
```

## Usage

このリポジトリはビルド不要の静的サイトです。ブラウザで `index.html` を開くか、ローカルサーバーを起動してアクセスします。

```bash
python3 -m http.server 8000
```

起動後、ブラウザで以下を開きます。

<http://localhost:8000/>

トップページから `Emotion` または `BlendShape` を選択してください。

## Notes

- Web カメラを使うため、ブラウザのカメラ権限を許可してください。
- `getUserMedia` は安全なコンテキストでのみ利用できます。ローカルでは `localhost` でのアクセスを推奨します。
- `emotion/` はリポジトリ内の `faceAPI/models/` を読み込みます。
- `blendshape/` は MediaPipe のライブラリとモデルを CDN から読み込むため、インターネット接続が必要です。

## Dependencies

- [face-api.js](https://github.com/justadudewhohacks/face-api.js)
- [MediaPipe Tasks Vision](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)
