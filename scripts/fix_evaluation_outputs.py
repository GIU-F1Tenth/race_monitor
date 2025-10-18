#!/usr/bin/env python3
"""
Small utility to fix evaluation outputs for an experiment folder.
- Removes unwanted CSV files (evaluation_summary.csv and timestamped race_results_*.csv)
- Renames race_evaluation_<ts>.json -> race_evaluation.json
- Generates a compact 'race_summary' file (JSON without extension) from the available summary CSV
- Produces a gzipped race_summary.rsum containing compact JSON

Usage: python3 fix_evaluation_outputs.py /path/to/experiment/results
"""
import sys
import os
import json
import gzip
import csv
from pathlib import Path


def read_csv_dict(filepath):
    data = []
    with open(filepath, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    return data


def parse_simple_race_results_csv(filepath):
    # parse the simple race_results CSV into a dict
    d = {}
    rows = read_csv_dict(filepath)
    i = 0
    # try to find key/value rows
    for row in rows:
        if not row:
            continue
        if len(row) == 2:
            key = row[0].strip()
            val = row[1].strip()
            d[key] = val
    # parse lap times
    lap_times = []
    for row in rows:
        if len(row) == 2 and row[0].isdigit():
            try:
                lap_times.append(float(row[1]))
            except Exception:
                pass
    if lap_times:
        d['lap_times'] = lap_times
    return d


def parse_race_summary_csv(filepath):
    # Attempt to parse the human-readable race_summary CSV into a structured dict
    d = {'race_metadata': {}, 'lap_statistics': {}, 'trajectory_statistics': {}, 'performance_metrics': {}}
    with open(filepath, newline='') as f:
        reader = csv.reader(f)
        section = None
        for row in reader:
            if not row:
                continue
            first = row[0].strip()
            if first.startswith('==='):
                if 'METADATA' in first:
                    section = 'race_metadata'
                elif 'LAP STATISTICS' in first:
                    section = 'lap_statistics'
                elif 'INDIVIDUAL LAP TIMES' in first:
                    section = 'individual_laps'
                elif 'TRAJECTORY STATISTICS' in first:
                    section = 'trajectory_statistics'
                elif 'PERFORMANCE METRICS' in first:
                    section = 'performance_metrics'
                else:
                    section = None
                continue
            # handle rows depending on section
            if section == 'individual_laps' and len(row) >= 2 and row[0].isdigit():
                d.setdefault('lap_times', []).append(float(row[1]))
            elif section in ['race_metadata', 'lap_statistics', 'trajectory_statistics', 'performance_metrics'] and len(row) >= 2:
                key = row[0].strip().lower().replace(' ', '_')
                val = row[1].strip()
                # try to convert numeric
                try:
                    if '.' in val or 'e' in val.lower():
                        v = float(val)
                    else:
                        v = int(val)
                    d[section][key] = v
                except Exception:
                    # keep string
                    d[section][key] = val
    return d


def main(results_dir: str):
    r = Path(results_dir)
    if not r.exists() or not r.is_dir():
        print('Provided path is not a directory')
        return 2

    # Remove evaluation_summary.csv if present
    eval_csv = r / 'evaluation_summary.csv'
    if eval_csv.exists():
        print(f'Removing {eval_csv}')
        eval_csv.unlink()

    # Remove timestamped race_results_*.csv
    for p in r.glob('race_results_*.csv'):
        print(f'Removing {p}')
        p.unlink()

    # Rename race_evaluation_*.json -> race_evaluation.json (stable)
    found_eval = list(r.glob('race_evaluation_*.json'))
    if found_eval:
        src = found_eval[0]
        dst = r / 'race_evaluation.json'
        print(f'Renaming {src} -> {dst}')
        # load and inspect
        try:
            with open(src, 'r') as f:
                data = json.load(f)
            # if file content seems to contain mostly zeros, attempt to enrich using other files

            def is_empty_or_zero(d):
                if not d:
                    return True

                def any_nonzero(x):
                    if x is None:
                        return False
                    if isinstance(x, (int, float)):
                        try:
                            return not (x == 0 or (isinstance(x, float) and (str(x) == 'nan')))
                        except Exception:
                            return True
                    if isinstance(x, str):
                        return x.strip() != ''
                    if isinstance(x, dict):
                        return any(any_nonzero(v) for v in x.values())
                    if isinstance(x, (list, tuple)):
                        return any(any_nonzero(v) for v in x)
                    return True
                return not any_nonzero(d)

            if is_empty_or_zero(data):
                # try to enrich from available files
                rr = list(r.glob('race_results.json'))
                if rr:
                    try:
                        with open(rr[0], 'r') as f:
                            loaded = json.load(f)
                        # take race_evaluation if present
                        if isinstance(loaded, dict) and 'race_evaluation' in loaded:
                            data = loaded['race_evaluation']
                        else:
                            data = loaded
                        print('Populated evaluation data from race_results.json')
                    except Exception:
                        pass
                else:
                    # try race_summary.csv
                    summary_csv = list(r.glob('race_summary_*.csv'))
                    if summary_csv:
                        try:
                            parsed = parse_race_summary_csv(summary_csv[0])
                            data = {
                                'metadata': parsed.get(
                                    'race_metadata', {}), 'performance_summary': parsed.get(
                                    'performance_metrics', {}), 'lap_times': parsed.get(
                                    'lap_times', [])}
                            print('Populated evaluation data from race_summary CSV')
                        except Exception:
                            pass

            # write stable file
            with open(dst, 'w') as f:
                json.dump(data, f, indent=2)
            # remove old file
            src.unlink()
        except Exception as e:
            print('Failed to process existing race_evaluation file:', e)

    # Convert race_summary_<controller>_*.csv to a compact 'race_summary' file (no extension)
    summary_candidates = list(r.glob('race_summary_*.csv'))
    if summary_candidates:
        src = summary_candidates[0]
        print(f'Converting {src} -> race_summary (compact JSON)')
        parsed = parse_race_summary_csv(src)
        out_path = r / 'race_summary'
        with open(out_path, 'w') as f:
            json.dump(parsed, f, indent=2)
        # also create gzipped .rsum
        rsum_path = r / 'race_summary.rsum'
        with gzip.open(rsum_path, 'wt', encoding='utf-8') as gz:
            json.dump(parsed, gz, separators=(',', ':'))
        # remove original csv
        try:
            src.unlink()
        except Exception:
            pass

    print('Cleanup finished')
    return 0


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: fix_evaluation_outputs.py /path/to/results')
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
