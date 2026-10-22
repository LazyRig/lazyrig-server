"""Compare generation persistence writes; temporary SQLite databases, no models.

Automatic checkpoints are disabled ONLY in these temporary benchmark databases
so retained WAL growth measures written bytes without file reuse hiding them.
Production checkpointing and durability settings are unchanged.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lazyrig.storage import Store


def measure(store_class, compact, updates, chunk_bytes):
    with tempfile.TemporaryDirectory() as directory:
        store = store_class(directory)
        try:
            store.db.execute('PRAGMA wal_autocheckpoint=0')
            prompt = 'p' * 32_768
            job = {'id': 'benchmark', 'status': 'running', 'phase': 'Generating response',
                   'model': 'fixture', 'created': 1, 'updated': 1, 'prompt': prompt,
                   'input_message': {'role': 'user', 'content': prompt}, 'request': {},
                   'text': '', 'content': '', 'reasoning_content': '', 'context': {}}
            store.save_job(job)
            wal = Path(directory) / 'state.sqlite3-wal'
            before = wal.stat().st_size
            counters = {'commits': 0, 'full_job_writes': 0}

            def trace(statement):
                if statement == 'COMMIT':
                    counters['commits'] += 1
                if statement.startswith('INSERT OR REPLACE INTO jobs'):
                    counters['full_job_writes'] += 1

            store.db.set_trace_callback(trace)
            start = time.perf_counter()
            for n in range(1, updates + 1):
                output = 'x' * (n * chunk_bytes)
                job.update(text=output, content=output, updated=n + 1,
                           context={'generated_tokens': n})
                event = {'text': output, 'content': output, 'reasoning_content': '',
                         'context': job['context'], 'stage': 'initial', 'final': False}
                if compact:
                    store.event(job['id'], 'text', job_update=job, **event)
                else:
                    store.save_job(job)
                    store.event(job['id'], 'text', **event)
            elapsed = time.perf_counter() - start
            store.db.set_trace_callback(None)
            growth = wal.stat().st_size - before
            final = store.job(job['id'])
            assert final == job, 'Live job differs from the last committed update'
            cursor, count = 0, 0
            while batch := store.events(job['id'], cursor):
                for event in batch:
                    count += 1
                    assert event['text'] == 'x' * (count * chunk_bytes)
                    assert event['content'] == event['text']
                cursor = batch[-1]['seq']
            assert count == updates
            job.update(status='completed', phase='Complete')
            store.save_job(job)
            result = {**counters, 'wal_bytes_written_during_stream': growth,
                      'stream_seconds': round(elapsed, 6), 'sqlite_synchronous': store.db.execute('PRAGMA synchronous').fetchone()[0],
                      'event_body_bytes': store.db.execute('SELECT SUM(length(CAST(body AS BLOB))) FROM events').fetchone()[0],
                      'output_bytes': len(final['text']), 'output_sha256': hashlib.sha256(final['text'].encode()).hexdigest(),
                      'replayed_updates': count}
        finally:
            store.close()
        result['database_bytes_after_clean_close'] = (Path(directory) / 'state.sqlite3').stat().st_size
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', type=Path, help='Optional previous server source root for exact baseline Store')
    parser.add_argument('--updates', type=int, default=1000)
    parser.add_argument('--chunk-bytes', type=int, default=64)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.updates < 1 or args.chunk_bytes < 1:
        parser.error('updates and chunk-bytes must be positive')
    before_class = Store
    if args.baseline_root:
        spec = importlib.util.spec_from_file_location('lazyrig._benchmark_baseline', args.baseline_root / 'lazyrig/storage.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        before_class = module.Store
    report = {'scope': 'Controlled SQLite write volume, no GPU/model throughput claim',
              'updates': args.updates, 'chunk_bytes': args.chunk_bytes,
              'baseline': 'previous Store and full-save/event loop' if args.baseline_root else 'current Store with previous full-save/event loop'}
    report['before'] = measure(before_class, False, args.updates, args.chunk_bytes)
    report['after'] = measure(Store, True, args.updates, args.chunk_bytes)
    assert report['before']['output_sha256'] == report['after']['output_sha256']
    report['wal_write_reduction_percent'] = round(100 * (1 - report['after']['wal_bytes_written_during_stream'] /
                                                         report['before']['wal_bytes_written_during_stream']), 2)
    rendered = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end='')


if __name__ == '__main__':
    main()
