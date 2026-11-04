"""Completion report joins explicit TRs and writes safe, readable XLSX cells."""
import csv
import json
from xml.etree import ElementTree as ET
import zipfile

import pytest

from fmri_rt_preproc.run_qc_report import collect_run_qc_rows, write_run_qc_workbook


def log(folder, name, header, rows):
    with (folder / name).open('w', newline='', encoding='utf-8') as stream:
        w = csv.writer(stream); w.writerow(header); w.writerows(rows)


def test_outer_join_preserves_warmup_missing_rows_and_source_conflicts(tmp_path):
    log(tmp_path, 'scores.csv', ['volume_idx', 'score_raw', 'reg_ready'], [[31, 2.5, 1], [33, -1, 1]])
    log(tmp_path, 'scores_z.csv', ['volume_idx', 'score_raw'], [[33, -.3]])
    log(tmp_path, 'fd_rt.csv', ['volume_idx', 'fd'], [[1, .1], [31, .2]])
    log(tmp_path, 'decoder_qc_rt.csv', ['volume_idx', 'reference_correlation', 'reference_coverage_fraction', 'correlation_status'], [[1, .97, 1, 'ok'], [2, 'nan', .9, 'incomplete_coverage'], [33, .92, 1, 'ok']])
    log(tmp_path, 'regression_status_rt.csv', ['volume_idx', 'fd', 'reg_ready'], [[31, .25, 1]])
    columns, rows, sources = collect_run_qc_rows(tmp_path)
    assert [r['volume_idx'] for r in rows] == [1, 2, 31, 33]
    assert rows[0]['reference_correlation'] == .97  # Warmup preserved.
    assert 'score_raw' not in rows[0]
    assert rows[1]['reference_correlation'] is None
    assert rows[1]['correlation_status'] == 'incomplete_coverage'
    assert rows[2]['fd'] == .2
    assert rows[2]['regression_status_rt.fd'] == .25
    assert rows[-1]['score_z'] == -.3
    assert all('joined_by_volume_idx' == s['status'] for s in sources)


def test_unindexed_log_is_skipped_and_invalid_explicit_tr_is_reported(tmp_path):
    log(tmp_path, 'scores.csv', ['score_raw'], [[1], [2]])
    log(tmp_path, 'fd_rt.csv', ['volume_idx', 'fd'], [['oops', .2], [9, .3]])
    _, rows, sources = collect_run_qc_rows(tmp_path)
    assert rows == [{'volume_idx': 9, 'fd': .3}]
    assert sources[0]['status'] == 'skipped_no_explicit_TR'
    assert sources[1]['invalid_TR_rows'] == 1


def test_identical_duplicate_tr_is_counted_conflicting_duplicate_rejected(tmp_path):
    log(tmp_path, 'fd_rt.csv', ['volume_idx', 'fd'], [[1, .1], [1, .1]])
    _, rows, sources = collect_run_qc_rows(tmp_path)
    assert len(rows) == 1 and sources[0]['identical_duplicate_rows'] == 1
    log(tmp_path, 'fd_rt.csv', ['volume_idx', 'fd'], [[1, .1], [1, .2]])
    with pytest.raises(ValueError, match='Conflicting duplicate TR 1'):
        write_run_qc_workbook(tmp_path)


def test_xlsx_is_atomic_valid_xml_and_untrusted_text_is_not_formula(tmp_path):
    log(tmp_path, 'decoder_qc_rt.csv', ['volume_idx', 'reference_correlation', 'reference_coverage_fraction', 'correlation_status'], [[1, .98, 1, 'ok'], [2, 'nan', .5, 'incomplete_coverage']])
    log(tmp_path, 'scores.csv', ['volume_idx', 'score_raw', 'event_type'], [[2, -.1, '=HYPERLINK("https://bad.invalid")\x00']])
    metadata = {'reference_kind': 'prepared_session_reference', 'reference_label': 'Fallback α', 'full_decoder_voxel_count': 10, 'fixed_qc_voxel_count': 7}
    (tmp_path / 'decoder_qc_reference.json').write_text(json.dumps(metadata), encoding='utf-8')
    path = write_run_qc_workbook(tmp_path)
    assert path == tmp_path / 'run_qc.xlsx'
    assert not (tmp_path / 'run_qc.xlsx.tmp').exists()
    ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if name.endswith('.xml') or name.endswith('.rels'):
                ET.fromstring(z.read(name))
        workbook = ET.fromstring(z.read('xl/workbook.xml'))
        assert [n.attrib['name'] for n in workbook.findall('s:sheets/s:sheet', ns)] == ['Summary', 'Volumes']
        sheet = ET.fromstring(z.read('xl/worksheets/sheet2.xml'))
        assert not sheet.findall('.//s:f', ns)  # Text cannot become an Excel formula.
        text = ''.join(sheet.itertext())
        assert '=HYPERLINK' in text and '\x00' not in text
        assert len(sheet.findall('s:sheetData/s:row', ns)) == 3
        assert sheet.find('s:autoFilter', ns) is not None
        summary = z.read('xl/worksheets/sheet1.xml').decode('utf-8')
        assert 'prepared_session_reference' in summary and 'Fallback α' in summary
    assert write_run_qc_workbook(tmp_path) == path  # Completion can regenerate it.


def test_no_explicit_tr_data_means_no_empty_workbook(tmp_path):
    assert write_run_qc_workbook(tmp_path) is None
    assert not (tmp_path / 'run_qc.xlsx').exists()


def test_actual_acquisition_speed_basename_is_joined_by_tr(tmp_path):
    log(tmp_path, 'acquisition_speed_rt.csv', ['volume_idx', 'watchdog_to_analysis_s'], [[4, .16], [2, .14]])
    _, rows, sources = collect_run_qc_rows(tmp_path)
    assert rows == [{'volume_idx': 2, 'watchdog_to_analysis_s': .14},
                    {'volume_idx': 4, 'watchdog_to_analysis_s': .16}]
    assert sources[0]['file'] == 'acquisition_speed_rt.csv'
