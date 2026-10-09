"""Conservative manifest gate; validates records, not physical provenance truth."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

REQUIRED = {'source','object_id','independence_group_id','identity_evidence','device_id',
            'split','permission_evidence','rgb_path','rgb_sha256','reference_path',
            'reference_kind','units','calibration_path','registration_evidence',
            'previously_evaluated','baseline_training_overlap'}


def audit(path):
    path = Path(path).resolve()
    errors = []
    groups = defaultdict(set)
    objects = defaultdict(set)
    images = defaultdict(set)
    seen_images = set()
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED - set(reader.fieldnames or [])
        rows = list(reader)
    if missing:
        errors.append('Missing columns: ' + ', '.join(sorted(missing)))
    if not rows:
        errors.append('Empty registry: no eligible objects or measurements.')
    for i, row in enumerate(rows, 2):
        def fail(message):
            errors.append(f'row {i}: {message}')
        if any(not row.get(k, '').strip() for k in REQUIRED):
            fail('All required fields must be nonempty; unknown is not verified.')
        split = row.get('split')
        if split not in {'calibration','development','test'}:
            fail('split must be calibration/development/test')
        group = row.get('independence_group_id', '')
        groups[group].add(split)
        objects[(row.get('source'), row.get('object_id'))].add(group)
        if row.get('units') != 'mm':
            fail('Reference must be explicitly converted and verified in mm.')
        if row.get('reference_kind') not in {'measured_depth','physical_size'}:
            fail('Unverified or synthetic reference not eligible for this physical validation.')
        if row.get('previously_evaluated') not in {'yes','no'}:
            fail('previously_evaluated must be yes/no')
        if row.get('baseline_training_overlap') not in {'yes','no','unknown'}:
            fail('baseline_training_overlap must be yes/no/unknown')
        if split == 'test' and (row.get('previously_evaluated') != 'no' or
                                row.get('baseline_training_overlap') != 'no'):
            fail('Fresh independent test requires no prior evaluation or baseline training overlap.')
        for column in ['rgb_path','reference_path','calibration_path']:
            value = row.get(column, '')
            file = Path(value)
            if not file.is_absolute():
                file = path.parent / file
            if not value or not file.is_file():
                fail(f'{column}: missing regular file')
                continue
            if column == 'rgb_path':
                digest = hashlib.sha256(file.read_bytes()).hexdigest()
                if digest != row.get('rgb_sha256'):
                    fail('RGB SHA256 mismatch')
                images[digest].add(split)
                if digest in seen_images:
                    fail('Duplicate RGB content; repeated rows are not independent frames.')
                seen_images.add(digest)
    for group, splits in groups.items():
        if len(splits) != 1:
            errors.append(f'Independent group {group!r} crosses splits: {sorted(map(str,splits))}')
    for obj, memberships in objects.items():
        if len(memberships) != 1:
            errors.append(f'Object {obj!r} assigned inconsistent independent groups.')
    for digest, splits in images.items():
        if len(splits) != 1:
            errors.append(f'Identical RGB {digest} crosses splits.')
    passed = not errors
    return {'passed': passed, 'rows': len(rows),
            'declared_test_groups': sum(s == {'test'} for s in groups.values()),
            'eligible_test_groups': sum(s == {'test'} for s in groups.values()) if passed else 0,
            'errors': errors,
            'limitation': 'Manifest/file gate only; identity, permission, reference and registration need source review.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('registry', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.registry)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.write_text(payload)
    print(payload)
    raise SystemExit(0 if result['passed'] else 2)
