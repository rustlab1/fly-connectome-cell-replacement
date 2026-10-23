#!/usr/bin/env python3
"""Restore MaleCNS v1.0 connectivity in the existing model's neuron order.

The original processed neuron table is read-only. This script filters the official
segment graph to its body IDs, records unsigned counts, and applies its stored
presynaptic signs. All reconstructed outputs are written under data/v5.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import scipy
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[2]
URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/"
    "flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather"
)
RAW_NAME = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
EXPECTED_BYTES = 1_051_241_946
EXPECTED_MD5 = "f30e9dcca25cfd021bf1e7b3d975599e"
EXPECTED_NEURONS = 165_122
EXPECTED_EDGES = 25_563_197
LEGACY_SYNAPSE_TOTAL = 124_025_040


def hashes(path: Path) -> dict:
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            sha.update(block)
            md5.update(block)
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha.hexdigest(),
        "md5": md5.hexdigest(),
    }


def ensure_raw(raw: Path, download: bool) -> dict:
    partial = raw.with_suffix(raw.suffix + ".part")
    if not raw.exists():
        if partial.exists() and partial.stat().st_size == EXPECTED_BYTES:
            candidate = hashes(partial)
            if candidate["md5"] != EXPECTED_MD5:
                raise ValueError("Completed partial download failed the official MD5")
            partial.replace(raw)
        elif download:
            raw.parent.mkdir(parents=True, exist_ok=True)
            with urlopen(URL, timeout=120) as response, partial.open("wb") as output:
                while block := response.read(8 * 1024 * 1024):
                    output.write(block)
            candidate = hashes(partial)
            if candidate["bytes"] != EXPECTED_BYTES or candidate["md5"] != EXPECTED_MD5:
                raise ValueError("Downloaded graph failed official size/MD5 checks")
            partial.replace(raw)
        else:
            raise FileNotFoundError(f"Missing {raw}. Use --download to retrieve it.")
    info = hashes(raw)
    if info["bytes"] != EXPECTED_BYTES or info["md5"] != EXPECTED_MD5:
        raise ValueError("Source graph does not match the verified official object")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    start = time.monotonic()
    raw = ROOT / "data/raw" / RAW_NAME
    out = ROOT / "data/v5"
    out.mkdir(parents=True, exist_ok=True)
    report = ROOT / "report/v5/data_provenance.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    source_info = ensure_raw(raw, args.download)
    neuron_path = ROOT / "data/processed/neurons.parquet"
    neuron_info_before = hashes(neuron_path)
    neurons = pd.read_parquet(neuron_path)
    if len(neurons) != EXPECTED_NEURONS:
        raise ValueError(f"Unexpected neuron count: {len(neurons)}")
    if not np.array_equal(neurons.index.to_numpy(), np.arange(len(neurons))):
        raise ValueError("Stored model indices are not contiguous and zero-based")
    if not neurons.bodyId.is_unique or not neurons.sign.isin([-1, 1]).all():
        raise ValueError("Neuron IDs or presynaptic signs are invalid")
    ids = pd.Index(neurons.bodyId.to_numpy())
    pre_chunks, post_chunks, weight_chunks = [], [], []
    retained_rows = raw_rows = unsigned_sum = 0
    parquet_path = out / "traced_connections.parquet"
    parquet_part = out / "traced_connections.parquet.part"
    table_schema = pa.schema([
        ("body_pre", pa.int64()), ("body_post", pa.int64()), ("weight", pa.int64())
    ])
    source = pa.memory_map(str(raw), "r")
    reader = ipc.open_file(source)
    print(f"Source schema: {reader.schema}", flush=True)
    schema_names = reader.schema.names
    required = ["body_pre", "body_post", "weight"]
    if not set(required).issubset(schema_names):
        raise ValueError(f"Unexpected source columns: {schema_names}")
    columns = [schema_names.index(name) for name in required]
    with pq.ParquetWriter(parquet_part, table_schema, compression="zstd") as writer:
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            raw_rows += batch.num_rows
            body_pre, body_post, weights = [
                batch.column(i).to_numpy(zero_copy_only=False) for i in columns
            ]
            pre = ids.get_indexer(body_pre)
            post = ids.get_indexer(body_post)
            keep = (pre >= 0) & (post >= 0)
            if not keep.any():
                continue
            w = weights[keep].astype(np.int64)
            if (w <= 0).any() or not np.array_equal(w, weights[keep]):
                raise ValueError("Retained weights are not positive integer counts")
            pre_chunks.append(pre[keep].astype(np.int32))
            post_chunks.append(post[keep].astype(np.int32))
            weight_chunks.append(w.astype(np.float32))
            retained_rows += len(w)
            unsigned_sum += int(w.sum(dtype=np.int64))
            writer.write_table(pa.Table.from_arrays([
                pa.array(body_pre[keep], type=pa.int64()),
                pa.array(body_post[keep], type=pa.int64()),
                pa.array(w, type=pa.int64()),
            ], schema=table_schema))
            if batch_index % 250 == 0:
                print(f"Batch {batch_index}/{reader.num_record_batches}: "
                      f"{retained_rows:,} traced edges", flush=True)
    source.close()
    if retained_rows != EXPECTED_EDGES:
        raise ValueError(f"Retained edge count {retained_rows:,} differs from {EXPECTED_EDGES:,}")
    pre = np.concatenate(pre_chunks)
    post = np.concatenate(post_chunks)
    unsigned_weights = np.concatenate(weight_chunks)
    del pre_chunks, post_chunks, weight_chunks
    unsigned_float32_sum = int(unsigned_weights.sum(dtype=np.float32))
    print(f"Before signs: {len(neurons):,} neurons, {retained_rows:,} edges, "
          f"{unsigned_sum:,} exact synapses; float32 sum {unsigned_float32_sum:,}", flush=True)
    sign = neurons.sign.to_numpy(dtype=np.int8)[pre]
    matrix = sp.csc_matrix((unsigned_weights * sign, (post, pre)),
                           shape=(len(neurons), len(neurons)), dtype=np.float32)
    matrix.sum_duplicates()
    matrix.sort_indices()
    exact_matrix_sum = int(np.abs(matrix.data).sum(dtype=np.float64))
    legacy_matrix_sum = int(np.abs(matrix.data).sum(dtype=np.float32))
    if matrix.nnz != EXPECTED_EDGES or exact_matrix_sum != unsigned_sum:
        raise ValueError("Sparse matrix does not preserve unsigned source counts")
    # Validate the column sign convention without allocating a second full matrix.
    for first in range(0, len(neurons), 4096):
        last = min(first + 4096, len(neurons))
        a, b = matrix.indptr[first], matrix.indptr[last]
        expected = np.repeat(neurons.sign.to_numpy()[first:last],
                             np.diff(matrix.indptr[first:last + 1]))
        if not np.array_equal(np.sign(matrix.data[a:b]), expected):
            raise ValueError("Matrix signs do not match presynaptic neurons")
    matrix_path = out / "W_csc.npz"
    matrix_part = out / "W_csc.part.npz"
    sp.save_npz(matrix_part, matrix)
    matrix_part.replace(matrix_path)
    parquet_part.replace(parquet_path)
    if hashes(neuron_path) != neuron_info_before:
        raise RuntimeError("The source neuron table changed during reconstruction")
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "verified",
        "dataset": "male-cns:v1.0; flat graph, synapse confidence >= 0.5",
        "official_download_page": "https://male-cns.janelia.org/download/",
        "official_source_url": URL,
        "official_generation": "1780494887545976",
        "official_last_modified": "2026-06-03T13:54:47Z",
        "official_md5": EXPECTED_MD5,
        "raw_source": source_info,
        "neuron_table": neuron_info_before,
        "neuron_order_sha256": hashlib.sha256(
            neurons.bodyId.to_numpy(dtype="<i8").tobytes()).hexdigest(),
        "source_graph_rows": raw_rows,
        "retained_neurons": len(neurons),
        "retained_edge_rows_before_signs": retained_rows,
        "retained_synapses_exact_integer_before_signs": unsigned_sum,
        "retained_synapses_float32_before_signs": unsigned_float32_sum,
        "legacy_reported_synapses": LEGACY_SYNAPSE_TOTAL,
        "matrix": {
            **hashes(matrix_path), "shape": list(matrix.shape), "format": "csc",
            "dtype": str(matrix.dtype), "nnz": int(matrix.nnz),
            "absolute_sum_exact_float64": exact_matrix_sum,
            "absolute_sum_legacy_float32": legacy_matrix_sum,
            "orientation": "rows = postsynaptic model index; columns = presynaptic model index",
            "sign_rule": "copy sign column from unchanged original neurons.parquet",
            "all_column_signs_verified": True,
        },
        "traced_connections": hashes(parquet_path),
        "source_neuron_table_unchanged": True,
        "runtime": {
            "executable": sys.executable, "python": platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "pyarrow": pa.__version__,
        },
        "script": hashes(Path(__file__).resolve()),
        "elapsed_seconds": round(time.monotonic() - start, 1),
    }
    if unsigned_sum != LEGACY_SYNAPSE_TOTAL:
        provenance["synapse_total_note"] = (
            "The exact unsigned integer total differs from the manuscript legacy total. "
            "The legacy script printed a float32 reduction, which may round. "
            "Both exact and legacy-style matrix reductions are recorded; no weights were adjusted."
        )
    report.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"matrix": str(matrix_path), "runtime": sys.executable,
                      "unsigned_synapses_exact": unsigned_sum,
                      "matrix_synapses_legacy_float32": legacy_matrix_sum,
                      "provenance": str(report)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
