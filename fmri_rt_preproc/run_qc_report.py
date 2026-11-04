"""Completion-only QC workbook export, with explicit TR joins and no new deps.

The XLSX is a view of existing CSV logs. It does not affect live processing,
decoder scores or censor flags. Unknown/missing TRs are never joined by row
position; conflicting duplicate TRs raise an error instead of hiding data.
"""
from pathlib import Path
from collections import Counter
import csv
import json
import math
import re
import zipfile
from xml.etree.ElementTree import Element, SubElement, tostring


CSV_SOURCES = (
    "scores.csv", "scores_z.csv", "decoder_qc_rt.csv", "fd_rt.csv",
    "dvars_rt.csv", "regression_status_rt.csv", "coverage_rt.csv",
    "speed_test.csv", "acquisition_speed_rt.csv", "acquisition_speed.csv",
    "biopac_timelag.csv",
)
_TR_KEYS = ("volume_idx", "TR", "tr", "volume")
_ILLEGAL_XML = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")
_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def _parse_tr(value):
    text = str(value).strip()
    if not re.fullmatch(r"[0-9]+", text):
        raise ValueError("TR must be an explicit nonnegative integer.")
    return int(text)


def _cell_value(value):
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value if math.isfinite(value) else None
    text = str(value)
    stripped = text.strip()
    if stripped.lower() in {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
        return None
    # Preserve text as an inline-string cell, including strings beginning '='.
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", stripped):
        number = float(stripped)
        if math.isfinite(number):
            return int(stripped) if re.fullmatch(r"[+-]?\d+", stripped) and abs(number) < 1e15 else number
        return None
    return _ILLEGAL_XML.sub("", text)[:32767]


def collect_run_qc_rows(run_dir, *, csv_names=CSV_SOURCES):
    """Outer-join supported logs by their explicit volume/TR column.

    The first occurrence of a field keeps its short name. Repeated field names
    from another source get a ``source.field`` name, preserving both observations
    even when their values disagree. ``scores_z.score_raw`` is named ``score_z``.
    """
    run_dir = Path(run_dir)
    merged, columns, sources = {}, ["volume_idx"], []
    for name in csv_names:
        path = run_dir / name
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames or []
            tr_key = next((k for k in _TR_KEYS if k in fields), None)
            if tr_key is None:
                sources.append({"file": name, "status": "skipped_no_explicit_TR", "rows": 0})
                continue
            prefix = Path(name).stem
            mapping = {}
            for field in fields:
                if field == tr_key:
                    continue
                proposed = "score_z" if name == "scores_z.csv" and field == "score_raw" else field
                if proposed in columns:
                    proposed = prefix + "." + field
                if proposed in columns:
                    raise ValueError("Duplicate QC column: " + proposed)
                mapping[field] = proposed; columns.append(proposed)
            seen, invalid, identical_duplicates = {}, 0, 0
            for row in reader:
                try:
                    tr = _parse_tr(row.get(tr_key))
                except ValueError:
                    invalid += 1
                    continue
                payload = {mapping[k]: _cell_value(row.get(k)) for k in mapping}
                if tr in seen:
                    if seen[tr] != payload:
                        raise ValueError(f"Conflicting duplicate TR {tr} in {path}.")
                    identical_duplicates += 1
                    continue
                seen[tr] = payload
                merged.setdefault(tr, {"volume_idx": tr}).update(payload)
            sources.append({"file": name, "status": "joined_by_" + tr_key, "rows": len(seen), "invalid_TR_rows": invalid, "identical_duplicate_rows": identical_duplicates})
    return columns, [merged[tr] for tr in sorted(merged)], sources


def _column(index):
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _xml(node):
    return tostring(node, encoding="utf-8", xml_declaration=True)


def _worksheet(rows, widths, *, freeze="A2", autofilter=False):
    sheet = Element("worksheet", xmlns=_NS)
    views = SubElement(sheet, "sheetViews"); view = SubElement(views, "sheetView", workbookViewId="0")
    SubElement(view, "pane", ySplit="1", topLeftCell=freeze, activePane="bottomLeft", state="frozen")
    cols = SubElement(sheet, "cols")
    for i, width in enumerate(widths, 1):
        SubElement(cols, "col", min=str(i), max=str(i), width=str(width), customWidth="1")
    data = SubElement(sheet, "sheetData")
    for r, values in enumerate(rows, 1):
        line = SubElement(data, "row", r=str(r))
        for c, value in enumerate(values, 1):
            value = _cell_value(value)
            if value is None:
                continue
            attrs = {"r": _column(c) + str(r)}
            if r == 1:
                attrs["s"] = "1"
            elif isinstance(value, float):
                attrs["s"] = "2"
            cell = SubElement(line, "c", **attrs)
            if isinstance(value, bool):
                cell.set("t", "b"); SubElement(cell, "v").text = "1" if value else "0"
            elif isinstance(value, (int, float)):
                SubElement(cell, "v").text = str(value)
            else:
                cell.set("t", "inlineStr")
                text = SubElement(SubElement(cell, "is"), "t", {"{http://www.w3.org/XML/1998/namespace}space": "preserve"})
                text.text = value
    if autofilter and rows:
        SubElement(sheet, "autoFilter", ref="A1:" + _column(len(rows[0])) + str(len(rows)))
    return _xml(sheet)


def _styles():
    root = Element("styleSheet", xmlns=_NS)
    fmts = SubElement(root, "numFmts", count="1"); SubElement(fmts, "numFmt", numFmtId="164", formatCode="0.0000")
    fonts = SubElement(root, "fonts", count="2")
    f = SubElement(fonts, "font"); SubElement(f, "sz", val="11"); SubElement(f, "name", val="Calibri")
    f = SubElement(fonts, "font"); SubElement(f, "b"); SubElement(f, "color", rgb="FFFFFFFF"); SubElement(f, "sz", val="11"); SubElement(f, "name", val="Calibri")
    fills = SubElement(root, "fills", count="3")
    SubElement(SubElement(fills, "fill"), "patternFill", patternType="none")
    SubElement(SubElement(fills, "fill"), "patternFill", patternType="gray125")
    fill = SubElement(SubElement(fills, "fill"), "patternFill", patternType="solid"); SubElement(fill, "fgColor", rgb="FF193B55"); SubElement(fill, "bgColor", indexed="64")
    borders = SubElement(root, "borders", count="1"); SubElement(borders, "border")
    base = SubElement(root, "cellStyleXfs", count="1"); SubElement(base, "xf", numFmtId="0", fontId="0", fillId="0", borderId="0")
    cells = SubElement(root, "cellXfs", count="3")
    SubElement(cells, "xf", numFmtId="0", fontId="0", fillId="0", borderId="0", xfId="0")
    SubElement(cells, "xf", numFmtId="0", fontId="1", fillId="2", borderId="0", xfId="0", applyFont="1", applyFill="1")
    SubElement(cells, "xf", numFmtId="164", fontId="0", fillId="0", borderId="0", xfId="0", applyNumberFormat="1")
    named = SubElement(root, "cellStyles", count="1"); SubElement(named, "cellStyle", name="Normal", xfId="0", builtinId="0")
    return _xml(root)


def write_run_qc_workbook(run_dir, *, metadata=None, output_path=None):
    """Write ``run_qc.xlsx`` atomically at completion; return None without TRs.

    All text uses XLSX inline-string cells, so metadata/CSV text is never executed
    as an Excel formula. NaN/Inf are blank numeric observations with their logged
    status columns retained. No writer package or Excel installation is required.
    """
    run_dir = Path(run_dir)
    columns, rows, sources = collect_run_qc_rows(run_dir)
    if not rows:
        return None
    if len(rows) >= 1048576 or len(columns) > 16384:
        raise ValueError("QC log exceeds XLSX worksheet dimensions.")
    info = {}
    for name in ("session_metadata.json", "decoder_qc_reference.json"):
        path = run_dir / name
        if path.is_file():
            info[name] = json.loads(path.read_text(encoding="utf-8"))
    if metadata:
        info["provided_metadata"] = metadata
    status = Counter(row.get("correlation_status") for row in rows if row.get("correlation_status"))
    summary = [["QC item", "Value"], ["Run directory", str(run_dir.resolve())], ["Volume/TR rows", len(rows)], ["First recorded TR", rows[0]["volume_idx"]], ["Last recorded TR", rows[-1]["volume_idx"]], ["Join rule", "Outer join on explicit volume_idx/TR only; missing observations remain blank."], ["Correlation stage", "Spatially aligned EPI before smoothing, cGLM and normalization; all logged TRs including warmup."], ["QC effect", "Diagnostic only; no effect on scoring, censoring, regression or masks."]]
    for key, count in sorted(status.items()):
        summary.append(["Correlation status: " + key, count])
    for source in sources:
        summary.append(["Source: " + source["file"], json.dumps(source, sort_keys=True)])
    for source, mapping in info.items():
        for key, value in mapping.items():
            summary.append([source + ": " + key, json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value])
    target = Path(output_path) if output_path is not None else run_dir / "run_qc.xlsx"
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".tmp")
    content = '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>'
    rels = '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>'
    workbook = '<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="' + _NS + '" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Summary" sheetId="1" r:id="rId1"/><sheet name="Volumes" sheetId="2" r:id="rId2"/></sheets></workbook>'
    workbook_rels = '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>'
    try:
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, text in (("[Content_Types].xml", content), ("_rels/.rels", rels), ("xl/workbook.xml", workbook), ("xl/_rels/workbook.xml.rels", workbook_rels)):
                archive.writestr(name, text)
            archive.writestr("xl/styles.xml", _styles())
            archive.writestr("xl/worksheets/sheet1.xml", _worksheet(summary, [52, 110]))
            table = [columns] + [[row.get(column) for column in columns] for row in rows]
            archive.writestr("xl/worksheets/sheet2.xml", _worksheet(table, [min(42, max(14, len(c) + 2)) for c in columns], autofilter=True))
        temp.replace(target)
    finally:
        if temp.exists():
            temp.unlink()
    return target
