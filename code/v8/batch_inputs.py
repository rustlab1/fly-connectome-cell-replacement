"""Read the immutable primary batch and separately recorded dose extension."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/v8'
STATE_FILES=['states.jsonl','typed_dose_states.jsonl','non_target_states.jsonl']


def read_inputs():
    primary=json.loads((OUT/'prespecified_protocol.json').read_text())
    amendment=json.loads((OUT/'typed_dose_amendment.json').read_text())
    assert hashlib.sha256((OUT/'prespecified_protocol.json').read_bytes()).hexdigest()==amendment['primary_protocol_sha256']
    specs=primary['primary_specs']+amendment['extension_specs']
    non_target=json.loads((OUT/'non_target_amendment.json').read_text())
    specs+=non_target['specs']
    for relative,expected in non_target['source_sha256'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected
    rows=[]
    for name in STATE_FILES:
        path=OUT/name
        if path.exists(): rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return primary,amendment,specs,rows
