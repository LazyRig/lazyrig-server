"""Read a saved placement report without loading a model or opening the server."""
import argparse
import json
from pathlib import Path
import re
import sqlite3


MAX_JOB_BYTES = 8 * 1024 * 1024


def read_report(database, prefix):
    if not isinstance(prefix, str) or not re.fullmatch(r'[a-f0-9]{8,32}', prefix):
        raise ValueError('Use the 8-character job ID from the terminal, or its full ID')
    path = Path(database).resolve(strict=True)
    # Never instantiate Store: it owns a write lock and performs recovery.
    connection = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=5)
    try:
        connection.execute('PRAGMA query_only=ON')
        # The ID range uses the jobs primary-key index, not the large event table.
        rows = connection.execute(
            'SELECT id FROM jobs WHERE id >= ? AND id < ? ORDER BY id LIMIT 2',
            (prefix, prefix + 'g')).fetchall()
        if not rows:
            raise ValueError('No saved job matches ' + prefix + '; check --database')
        if len(rows) != 1:
            raise ValueError('Several jobs match that prefix; use a longer job ID')
        jid = rows[0][0]
        body = connection.execute('SELECT substr(body,1,?) FROM jobs WHERE id=?',
                                  (MAX_JOB_BYTES + 1, jid)).fetchone()[0]
        if len(body) > MAX_JOB_BYTES:
            raise ValueError('Saved job is too large for this bounded diagnostic reader')
        job = json.loads(body)
    finally:
        connection.close()
    reports = [{'kind': event['kind'], 'target': event.get('target'), 'preflight': event['preflight']}
               for event in job.get('placement_events', [])
               if event.get('kind') in {'placement_rejected', 'placement_selected'} and event.get('preflight')]
    if not reports:
        raise ValueError('This job contains no saved manual-split preflight report')
    # Exclude prompts, messages, requests, filesystem paths and API credentials.
    return {'job_id': jid, 'model': job.get('model'), 'status': job.get('status'),
            'error_code': job.get('error_code'), 'placement_reports': reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True, help='8-character terminal job ID or full ID')
    parser.add_argument('--database', type=Path, default=Path('server-data/state.sqlite3'))
    args = parser.parse_args()
    try:
        report = read_report(args.database, args.job)
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.exit(1, str(exc) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
