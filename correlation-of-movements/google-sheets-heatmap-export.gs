const SPREADSHEET_ID = "1Fv-YGBgFj2W0OcB8acxQ797VIbtDIqoprwCjCHi6azI";
const OUTPUT_COLUMN_WIDTH = 146;

function doGet() {
  return jsonResponse({ ok: true, message: "Heatmap export endpoint is ready." });
}

function doPost(e) {
  try {
    const payload = parsePayload(e);
    const spreadsheet = SpreadsheetApp.openById(SPREADSHEET_ID);
    const sheet = createHeatmapSheet(spreadsheet);

    writeHeatmap(sheet, payload);
    return jsonResponse({ ok: true, sheetName: sheet.getName(), writtenAt: new Date().toISOString() });
  } catch (error) {
    return jsonResponse({ ok: false, error: error.message });
  }
}

function parsePayload(e) {
  const body = e?.postData?.contents || e?.parameter?.payload || "";
  if (!body) throw new Error("payload is empty");

  const payload = JSON.parse(body);
  if (!Array.isArray(payload.labels) || !Array.isArray(payload.matrix)) {
    throw new Error("payload.labels and payload.matrix are required");
  }
  if (payload.matrix.length !== payload.labels.length || payload.matrix.some((row) => row.length !== payload.labels.length)) {
    throw new Error("matrix size does not match labels");
  }
  return payload;
}

function createHeatmapSheet(spreadsheet) {
  const nameRoot = Utilities.formatDate(new Date(), Session.getScriptTimeZone(), "yyyyMMdd_HHmmss");
  let sheetName = nameRoot;
  let suffix = 2;

  while (spreadsheet.getSheetByName(sheetName)) {
    const suffixText = `_${suffix++}`;
    sheetName = `${nameRoot.slice(0, 100 - suffixText.length)}${suffixText}`;
  }
  return spreadsheet.insertSheet(sheetName);
}

function writeHeatmap(sheet, payload) {
  const labels = payload.labels;
  const names = Array.isArray(payload.names) ? payload.names : labels;
  const matrix = payload.matrix;
  const size = labels.length;
  const exportedAt = payload.exportedAt ? new Date(payload.exportedAt) : new Date();

  ensureSheetSize(sheet, size + 6, size + 4);
  sheet.getRange(1, 1, 4, 2).setValues([
    ["Correlation heatmap", ""],
    ["Exported at", exportedAt],
    ["Frames", Number(payload.frames || 0)],
    ["Excluded variables", Number(payload.excluded || 0)]
  ]);
  sheet.getRange(2, 2).setNumberFormat("yyyy-mm-dd hh:mm:ss");

  const table = [
    ["", ...labels],
    ...labels.map((label, row) => [label, ...matrix[row].map((value, column) => (
      row === column ? '=SPARKLINE({1,0})' : Number(value)
    ))])
  ];
  const tableRange = sheet.getRange(6, 1, size + 1, size + 1);
  tableRange.setValues(table);
  tableRange.setHorizontalAlignment("center");
  tableRange.setVerticalAlignment("middle");
  sheet.getRange(7, 2, size, size).setNumberFormat("0.00");
  sheet.getRange(7, 2, size, size).setFontWeight("normal");
  for (let i = 0; i < size; i++) {
    sheet.getRange(7 + i, 2 + i).setNumberFormat("General").setFontWeight("normal");
  }

  sheet.getRange(6, 1, 1, size + 1).setFontWeight("bold").setBackground("#dfe7f1");
  sheet.getRange(6, 1, size + 1, 1).setFontWeight("bold").setBackground("#dfe7f1");
  sheet.getRange(7, 2, size, size).setBackgrounds(matrix.map((row, y) => (
    row.map((value, x) => x === y ? "#f4f7fb" : colorForCorrelation(value))
  )));

  const originalNameRows = [["Display label", "Blendshape name"], ...labels.map((label, index) => [label, names[index] || label])];
  sheet.getRange(6, size + 3, originalNameRows.length, 2).setValues(originalNameRows);
  sheet.getRange(6, size + 3, 1, 2).setFontWeight("bold").setBackground("#dfe7f1");

  sheet.setFrozenRows(6);
  sheet.setFrozenColumns(1);
  sheet.setColumnWidths(1, size + 4, OUTPUT_COLUMN_WIDTH);
}

function ensureSheetSize(sheet, minRows, minColumns) {
  if (sheet.getMaxRows() < minRows) {
    sheet.insertRowsAfter(sheet.getMaxRows(), minRows - sheet.getMaxRows());
  }
  if (sheet.getMaxColumns() < minColumns) {
    sheet.insertColumnsAfter(sheet.getMaxColumns(), minColumns - sheet.getMaxColumns());
  }
}

function colorForCorrelation(value) {
  const v = Math.min(1, Math.max(-1, Number(value || 0)));
  const t = Math.abs(v);
  const end = v >= 0 ? [190, 38, 51] : [42, 96, 180];
  return `#${end.map((channel) => {
    const mixed = Math.round(255 + (channel - 255) * t);
    return mixed.toString(16).padStart(2, "0");
  }).join("")}`;
}

function jsonResponse(data) {
  return ContentService
    .createTextOutput(JSON.stringify(data))
    .setMimeType(ContentService.MimeType.JSON);
}
