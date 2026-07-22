"""
Export a single combined CSV dataset from all per-event-log runs files in the tar cache.

Each row = one (event_log, model_id, trace_id, aligner) instance with:
- event log name
- timing data (all recorded metrics)
- all 49 named feature columns (model structure, trace, interaction, token replay)
- is_best_aligner: True if this aligner had the lowest mean time for this combination
"""

import tarfile
import csv
import io
import os
import numpy as np
from urllib.parse import unquote

TAR_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "pm_ws25_cache.tar.gz")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "combined_dataset.csv")

# Feature names from CompositeFeatureExtractor (model + trace + interaction + token_replay)
FEATURE_NAMES = [
    # Model structure features (37)
    'model_n_transitions', 'model_n_places', 'model_n_arcs',
    'model_n_inv_transition', 'model_n_dup_transition', 'model_n_uniq_transition',
    'model_n_and_split', 'model_n_xor_split',
    'model_inv_tran_in_deg_mean', 'model_inv_tran_in_deg_std',
    'model_inv_tran_out_deg_mean', 'model_inv_tran_out_deg_std',
    'model_uniq_tran_in_deg_mean', 'model_uniq_tran_in_deg_std',
    'model_uniq_tran_out_deg_mean', 'model_uniq_tran_out_deg_std',
    'model_dup_tran_in_deg_mean', 'model_dup_tran_in_deg_std',
    'model_dup_tran_out_deg_mean', 'model_dup_tran_out_deg_std',
    'model_place_in_deg_mean', 'model_place_in_deg_std',
    'model_place_out_deg_mean', 'model_place_out_deg_std',
    'model_tran_in_deg_mean', 'model_tran_in_deg_std',
    'model_tran_out_deg_mean', 'model_tran_out_deg_std',
    'model_and_split_avg_out_deg', 'model_and_split_max_out_deg', 'model_and_split_out_deg_std',
    'model_xor_split_avg_out_deg', 'model_xor_split_max_out_deg', 'model_xor_split_out_deg_std',
    'model_density_arcs_per_transition', 'model_density_arcs_per_transition_plus_places',
    'model_density_arcs_per_place',
    # Trace features (3)
    'trace_length', 'trace_activity_repeat_mean', 'trace_activity_repeat_std',
    # Interaction features (3)
    'interaction_n_activity_present_in_model', 'interaction_n_activity_not_in_model',
    'interaction_activity_coverage_ratio',
    # Token replay features (6)
    'token_replay_trace_is_fit', 'token_replay_trace_fitness',
    'token_replay_missing_tokens', 'token_replay_consumed_tokens',
    'token_replay_remaining_tokens', 'token_replay_produced_tokens',
]

TIMING_COLS = ['time_total_mean', 'time_total_std', 'time_total_median', 'time_search_mean', 'time_lp_mean']

COLUMNS = (
    ['event_log', 'model_id', 'trace_id', 'aligner', 'is_best_aligner', 'item_id', 'combination_id']
    + TIMING_COLS
    + FEATURE_NAMES
)


def build_uuid_to_name():
    mapping = {}
    for uuid in os.listdir(DATA_DIR):
        data_path = os.path.join(DATA_DIR, uuid)
        if not os.path.isdir(data_path):
            continue
        candidates = [
            f for f in os.listdir(data_path)
            if not f.startswith('.') and f not in ('DATA.xml', 'DATA1.xml', 'readme.txt', 'README.txt')
            and not f.endswith('.json')
        ]
        if candidates:
            name = os.path.splitext(candidates[0])[0]
            mapping[uuid] = unquote(name)
        else:
            mapping[uuid] = uuid
    return mapping


def parse_feature_vector(fv_str):
    """Parse numpy array string like '[ 1.0  2.0  3.0\n 4.0 ]' into list of floats."""
    try:
        cleaned = fv_str.replace('\n', ' ').replace('[', '').replace(']', '')
        return [float(x) for x in cleaned.split() if x]
    except Exception:
        return [float('nan')] * len(FEATURE_NAMES)


def load_runs_from_text(text_wrapper):
    """Load all rows from a runs CSV, return list of dicts and set of best item_ids."""
    rows = []
    reader = csv.DictReader(text_wrapper)
    for row in reader:
        rows.append(row)

    # Determine best aligner per combination_id (lowest time_total_mean)
    best_item_ids = set()
    by_combination = {}
    for row in rows:
        cid = row['combination_id']
        try:
            t = float(row['time_total_mean'])
        except (ValueError, KeyError):
            t = float('inf')
        if cid not in by_combination or t < by_combination[cid][0]:
            by_combination[cid] = (t, row['item_id'])
    for _, item_id in by_combination.values():
        best_item_ids.add(item_id)

    return rows, best_item_ids


def main():
    uuid_to_name = build_uuid_to_name()
    print(f"Found {len(uuid_to_name)} event log mappings")

    total_rows = 0
    with open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=COLUMNS)
        writer.writeheader()

        with tarfile.open(TAR_PATH, 'r:gz') as tar:
            for member in tar.getmembers():
                if not member.name.endswith('.runs.csv'):
                    continue

                basename = os.path.basename(member.name)
                uuid = basename.replace('.runs.csv', '')
                event_log_name = uuid_to_name.get(uuid, uuid)
                print(f"Processing: {event_log_name} ({uuid})")

                f = tar.extractfile(member)
                if f is None:
                    continue

                text = io.TextIOWrapper(f, encoding='utf-8')
                rows, best_item_ids = load_runs_from_text(text)

                for row in rows:
                    fv = parse_feature_vector(row.get('feature_vector', ''))
                    if len(fv) != len(FEATURE_NAMES):
                        fv = fv[:len(FEATURE_NAMES)] + [float('nan')] * max(0, len(FEATURE_NAMES) - len(fv))

                    out_row = {
                        'event_log': event_log_name,
                        'model_id': row['model_id'],
                        'trace_id': row['trace_id'],
                        'aligner': row['aligner'],
                        'is_best_aligner': row['item_id'] in best_item_ids,
                        'item_id': row['item_id'],
                        'combination_id': row['combination_id'],
                    }
                    for col in TIMING_COLS:
                        out_row[col] = row.get(col, '')
                    for i, fname in enumerate(FEATURE_NAMES):
                        out_row[fname] = fv[i]

                    writer.writerow(out_row)
                    total_rows += 1

    print(f"\nDone. Total rows: {total_rows:,}")
    print(f"Output: {os.path.abspath(OUTPUT_PATH)}")
    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"File size: {size_mb:.1f} MB")
    print(f"\nColumns ({len(COLUMNS)} total):")
    print(f"  - Identifiers: event_log, model_id, trace_id, aligner, is_best_aligner, item_id, combination_id")
    print(f"  - Timing: {TIMING_COLS}")
    print(f"  - Features: {len(FEATURE_NAMES)} named columns (model structure + trace + interaction + token replay)")


if __name__ == '__main__':
    main()
