# 表情測定証明証 HTML/CSS

- `index.html`：証明証本体
- `styles.css`：レイアウトと印刷設定
- `assets/logo.png`：ロゴマーク
- `assets/seal.png`：配置用に濃度を微調整した印鑑
- `assets/seal-source.png`：アップロードされた印鑑の原画像

## 表示

`index.html` をブラウザで開いてください。

## L版で印刷

ブラウザの印刷画面での指定

- 用紙：L版（127 × 89 mm）
- 向き：横
- 余白：なし
- 拡大縮小：100% / 実際のサイズ
- 背景グラフィック：オン

CSS側にも `@page { size: 127mm 89mm; margin: 0; }` を設定しています。
