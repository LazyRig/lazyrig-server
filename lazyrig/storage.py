"""Durable conversations, compact replayable events and exclusive state ownership."""
from collections import OrderedDict
from functools import wraps
import json
import logging
import re
import shutil
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from .idempotency import IdempotencyConflict
from .storage_lock import DirectoryLock

TERMINAL = {'completed', 'failed', 'cancelled', 'interrupted'}
STREAM_FIELDS = ('text', 'content', 'reasoning_content')
STREAM_JOB_EVENTS = {'text', 'progress', 'enhancement_progress'}
STREAM_JOB_FIELDS = {'status', 'phase', 'updated', 'context', 'timings', 'progress',
                     'thinking_open', 'reasoning_incomplete'}
SUMMARY_FIELDS = {'id', 'kind', 'model', 'status', 'phase', 'created', 'updated',
                  'context', 'timings', 'progress', 'error', 'artifacts', 'placement_plan', 'placement_attempts'}
LOGGER = logging.getLogger('lazyrig.runtime')


def text_events(alias=''):
    """Legacy rows may intentionally have no kind metadata; never backfill them at startup."""
    prefix = alias + '.' if alias else ''
    return (f"({prefix}kind='text' OR ({prefix}kind='' AND {prefix}encoding=1 "
            f"AND json_extract({prefix}body,'$.kind')='text'))")


def new_id():
    return uuid.uuid4().hex


def storage_operation(method):
    """A failed durable operation must make readiness/admission fail closed."""
    @wraps(method)
    def guarded(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except (sqlite3.Error, OSError, json.JSONDecodeError) as exc:
            with self.lock:
                self._storage_error = str(exc)
                self._health_checked = time.time()
            raise
    return guarded


def job_summary(job):
    result = {**{k: v for k, v in job.items() if k in SUMMARY_FIELDS},
            'prompt': job.get('prompt', '')[:160],
            'request': {k: v for k, v in job.get('request', {}).items()
                        if k in {'max_tokens', 'width', 'height'}}}
    if 'placement_attempts' in result:
        result['placement_attempts'] = [
            {k: (v[:600] if k == 'reason' else v) for k, v in a.items()
             if k in {'attempt', 'target', 'label', 'status', 'reason_code', 'reason'}}
            for a in result['placement_attempts'][:32]]
    return result


class Store:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = None
        self._closed = False
        self._storage_error = None
        self._health_checked = time.time()
        self._write_heads = OrderedDict()
        self._replay_cache = OrderedDict()
        # Recovery below changes live job states. Acquire ownership first, even
        # when a second instance has a different listening port.
        self._owner = DirectoryLock(self.directory)
        try:
            (self.directory / 'artifacts').mkdir(exist_ok=True)
            database = self.directory / 'state.sqlite3'
            size = database.stat().st_size if database.exists() else 0
            LOGGER.info('Startup: opening state database (%.1f GiB).', size / 1024**3)
            self.db = sqlite3.connect(database, check_same_thread=False)
            self.db.execute('PRAGMA journal_mode=WAL')
            self.db.executescript('''
              CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, title TEXT, created REAL);
              CREATE TABLE IF NOT EXISTS messages(seq INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT, body TEXT);
              CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, body TEXT);
              CREATE TABLE IF NOT EXISTS job_streams(job TEXT PRIMARY KEY, body TEXT NOT NULL, text_seq INTEGER);
              CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT, job TEXT, body TEXT);
              CREATE INDEX IF NOT EXISTS events_job_seq ON events(job,seq);
              CREATE TABLE IF NOT EXISTS image_request_keys(key_hash TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, job TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS store_metadata(key TEXT PRIMARY KEY, value TEXT);
            ''')
            self._migrate()
            LOGGER.info('Startup: recovering unfinished jobs...')
            for (jid,) in self.db.execute("SELECT id FROM jobs WHERE status NOT IN ('completed','failed','cancelled','interrupted')").fetchall():
                # Restore the last committed stream update before marking an
                # interrupted job; full snapshots are no longer saved per chunk.
                job = self.job(jid)
                if job.get('cancel_requested'):
                    job.update(status='cancelled', phase='Cancelled',
                               error='Cancellation was requested before server restart. This job was not replayed.')
                else:
                    job.update(status='interrupted', error='Server restarted. This job was not replayed automatically.')
                self.save_job(job)
            LOGGER.info('Startup: database ready.')
        except BaseException:
            self.close()
            raise

    def _migrate(self):
        """Add metadata without scanning or rewriting the historical event table.

        Old cumulative events can occupy tens of GiB. Extracting JSON from every
        row and building secondary event indexes here made startup silent and
        disk-bound. Leave legacy payloads and metadata untouched; readers handle
        both layouts, including an interrupted first-beta migration.
        """
        LOGGER.info('Startup: checking database schema...')
        job_columns = {r[1] for r in self.db.execute('PRAGMA table_info(jobs)')}
        for column, type_ in (('summary', 'TEXT'), ('status', 'TEXT'), ('updated', 'REAL')):
            if column not in job_columns:
                self.db.execute(f'ALTER TABLE jobs ADD COLUMN {column} {type_}')
        event_columns = {r[1] for r in self.db.execute('PRAGMA table_info(events)')}
        for column, declaration in (('encoding', 'INTEGER NOT NULL DEFAULT 1'),
                                    ('checkpoint', 'INTEGER NOT NULL DEFAULT 0'),
                                    ('kind', "TEXT NOT NULL DEFAULT ''")):
            if column not in event_columns:
                self.db.execute(f'ALTER TABLE events ADD COLUMN {column} {declaration}')
        # Updates leave existing rowids/order, event sequence IDs and retry keys
        # intact. Iteration avoids retaining all old job bodies in Python memory.
        migrated = 0
        last_notice = time.monotonic()
        for jid, body in self.db.execute('SELECT id,body FROM jobs WHERE summary IS NULL OR status IS NULL'):
            job = json.loads(body)
            self.db.execute('UPDATE jobs SET summary=?,status=?,updated=? WHERE id=?',
                            (json.dumps(job_summary(job)), job['status'], job.get('updated', job.get('created', 0)), jid))
            migrated += 1
            if time.monotonic() - last_notice >= 2:
                LOGGER.info('Startup: prepared %s job summaries; historical events remain untouched.', migrated)
                last_notice = time.monotonic()
        self.db.execute('CREATE INDEX IF NOT EXISTS jobs_status_updated ON jobs(status,updated)')
        # Empty databases can get these indexes cheaply. Upgrades keep existing
        # indexes, but never build a full-table secondary index over old payloads.
        if self.db.execute('SELECT 1 FROM events LIMIT 1').fetchone() is None:
            self.db.execute('CREATE INDEX IF NOT EXISTS events_checkpoint ON events(job,checkpoint,seq)')
            self.db.execute('CREATE INDEX IF NOT EXISTS events_job_kind_seq ON events(job,kind,seq)')
        self.db.execute("INSERT OR REPLACE INTO store_metadata VALUES('schema_version','4')")
        self.db.commit()

    def health(self, probe=False):
        """Latched health; only an actual successful write probe clears failure."""
        with self.lock:
            if self._closed:
                return {'ok': False, 'writable': False, 'error': 'Storage is closed',
                        'checked_at': self._health_checked, 'free_disk_bytes': None}
            if probe:
                try:
                    with self.db:
                        self.db.execute("INSERT OR REPLACE INTO store_metadata VALUES('write_probe',?)", (str(time.time()),))
                    self._storage_error = None
                except (sqlite3.Error, OSError) as exc:
                    self._storage_error = str(exc)
                self._health_checked = time.time()
            try:
                free_disk_bytes = shutil.disk_usage(self.directory).free
            except OSError as exc:
                free_disk_bytes = None
                self._storage_error = str(exc)
                self._health_checked = time.time()
            ok = self._storage_error is None
            return {'ok': ok, 'writable': ok, 'error': self._storage_error,
                    'checked_at': self._health_checked, 'free_disk_bytes': free_disk_bytes}

    @storage_operation
    def session(self, title='New conversation'):
        sid = new_id()
        with self.lock, self.db:
            self.db.execute('INSERT INTO sessions VALUES(?,?,?)', (sid, title[:80], time.time()))
        return sid

    @storage_operation
    def sessions(self):
        with self.lock:
            return [dict(zip(('id', 'title', 'created'), r)) for r in
                    self.db.execute('SELECT * FROM sessions ORDER BY created DESC')]

    @storage_operation
    def messages(self, sid):
        with self.lock:
            if not self.db.execute('SELECT 1 FROM sessions WHERE id=?', (sid,)).fetchone():
                raise KeyError('Conversation not found')
            return [json.loads(r[0]) for r in self.db.execute('SELECT body FROM messages WHERE session=? ORDER BY seq', (sid,))]

    @storage_operation
    def append(self, sid, message):
        with self.lock, self.db:
            self.db.execute('INSERT INTO messages(session,body) VALUES(?,?)', (sid, json.dumps(message)))

    @storage_operation
    def save_job(self, job, retry_identity=None):
        values = (job['id'], json.dumps(job), json.dumps(job_summary(job)), job['status'],
                  job.get('updated', job.get('created', 0)))
        with self.lock, self.db:
            if retry_identity is None:
                self.db.execute('INSERT OR REPLACE INTO jobs(id,body,summary,status,updated) VALUES(?,?,?,?,?)', values)
            else:
                try:
                    self.db.execute('INSERT INTO jobs(id,body,summary,status,updated) VALUES(?,?,?,?,?)', values)
                    self.db.execute('INSERT INTO image_request_keys VALUES(?,?,?)', (*retry_identity, job['id']))
                except sqlite3.IntegrityError as exc:
                    prior = self.db.execute('SELECT job FROM image_request_keys WHERE key_hash=?', (retry_identity[0],)).fetchone()
                    raise IdempotencyConflict(prior[0] if prior else job['id']) from exc
            # Lifecycle snapshots supersede stream state in the same commit.
            self.db.execute('DELETE FROM job_streams WHERE job=?', (job['id'],))

    @storage_operation
    def inference_request(self, identity):
        with self.lock:
            row = self.db.execute('SELECT fingerprint,job FROM image_request_keys WHERE key_hash=?', (identity[0],)).fetchone()
            if row is None:
                return None
            if row[0] != identity[1]:
                raise IdempotencyConflict(row[1])
            return self.job(row[1])

    image_request = inference_request

    @storage_operation
    def job(self, jid):
        with self.lock:
            row = self.db.execute('SELECT body FROM jobs WHERE id=?', (jid,)).fetchone()
            if not row:
                raise KeyError('Job not found')
            progress = self.db.execute('SELECT body,text_seq FROM job_streams WHERE job=?', (jid,)).fetchone()
            return self._project_job(json.loads(row[0]), progress)

    def _project_job(self, job, progress, summary=False):
        """Overlay committed small updates without rewriting the full job blob."""
        if progress is None:
            return job
        body, text_seq = progress
        updates = json.loads(body)
        job.update({k: v for k, v in updates.items() if not summary or k in SUMMARY_FIELDS})
        if text_seq is not None and not summary:
            head = self._write_heads.get(job['id'])
            if head is not None and head[2] == text_seq:
                state = head[0]
            else:
                state = self._replay_cache.get((job['id'], text_seq))
                if state is None:
                    state = self._state_at(job['id'], text_seq)
                    self._remember_replay(job['id'], text_seq, state)
            job.update(state)
        return job

    def _stream_rows(self):
        # Only jobs with unsnapshotted progress have a row, not all past jobs.
        return {jid: (body, seq) for jid, body, seq in
                self.db.execute('SELECT job,body,text_seq FROM job_streams')}

    def _recent_jobs(self, column, progress):
        # Full snapshots no longer move to the newest row on every chunk. Keep
        # streaming jobs visible even after 200 newer jobs have been saved.
        seen = set()
        for jid in progress:
            row = self.db.execute(f'SELECT {column} FROM jobs WHERE id=?', (jid,)).fetchone()
            if row is not None:
                seen.add(jid)
                yield json.loads(row[0])
        for (body,) in self.db.execute(f'SELECT {column} FROM jobs ORDER BY rowid DESC'):
            job = json.loads(body)
            if job['id'] not in seen:
                yield job

    @storage_operation
    def jobs(self, summary=False):
        with self.lock:
            column = 'summary' if summary else 'body'
            progress = self._stream_rows()
            result = []
            for job in self._recent_jobs(column, progress):
                result.append(self._project_job(job, progress.get(job['id']), summary))
                if len(result) == 200:
                    break
            return sorted(result, key=lambda job: job.get('updated', job.get('created', 0)), reverse=True) if progress else result

    @storage_operation
    def image_jobs(self, model=None):
        with self.lock:
            result = []
            progress = self._stream_rows()
            for job in self._recent_jobs('body', progress):
                image = job.get('kind') == 'image' or (not job.get('kind') and
                        {'width', 'steps'} <= set(job.get('request', {})) and 'max_tokens' not in job['request'])
                if image and (model is None or job['model'] == model):
                    result.append(self._project_job(job, progress.get(job['id'])))
                    if len(result) == 200:
                        break
            return sorted(result, key=lambda job: job.get('updated', job.get('created', 0)), reverse=True) if progress else result

    @storage_operation
    def artifacts(self, job_lookup=None):
        """List existing image files, including outputs without a job record."""
        lookup = job_lookup or self.job
        candidates = []
        for path in (self.directory / 'artifacts').iterdir():
            if not re.fullmatch(r'[a-f0-9]{32}\.(png|svg)', path.name) or path.is_symlink():
                continue
            try:
                if path.is_file():
                    info = path.stat()
                    candidates.append((info.st_mtime, path, info.st_size))
            except FileNotFoundError:
                continue
        result = []
        for modified, path, size in sorted(candidates, reverse=True):
            try:
                job = lookup(path.stem)
            except KeyError:
                job = None
            if job and not job.get('artifacts'):
                continue
            result.append({'name': path.name, 'url': '/artifacts/' + path.name,
                           'bytes': size, 'modified': modified, 'job_id': job['id'] if job else None,
                           'model': job['model'] if job else None, 'prompt': job['prompt'] if job else None,
                           'job_status': job['status'] if job else None,
                           **({'x_durable': job.get('durable', True), 'x_storage_warning': job.get('storage_error')} if job else {})})
            if len(result) == 200:
                break
        return result

    @staticmethod
    def _decode(body, encoding, state):
        record = json.loads(body)
        if encoding == 1:
            if record.get('kind') == 'text':
                state.update({k: record[k] for k in STREAM_FIELDS if isinstance(record.get(k), str)})
            return record
        if encoding != 2:
            raise sqlite3.DatabaseError('Unsupported stored event encoding')
        item = record['item']
        if 'state' in record:
            state.clear()
            state.update(record['state'])
            item.update({key: state[key] for key in record['fields']})
        else:
            for key, (keep, suffix) in record['fields'].items():
                previous = state.get(key, '')
                if not isinstance(keep, int) or not 0 <= keep <= len(previous):
                    raise sqlite3.DatabaseError('Invalid streamed event prefix')
                state[key] = previous[:keep] + suffix
                item[key] = state[key]
        return item

    def _state_at(self, jid, through):
        # Walk only this job, backwards to a compact checkpoint or a complete
        # legacy cumulative text event. The original (job,seq) index suffices.
        # Collapse partial legacy snapshots instead of retaining their repeated
        # full strings in a reconstruction buffer.
        state, legacy, pending = {}, {}, []
        rows = self.db.execute('SELECT body,encoding FROM events WHERE job=? AND seq<=? ORDER BY seq DESC',
                               (jid, through))
        for body, encoding in rows:
            record = json.loads(body)
            if encoding == 1:
                if record.get('kind') == 'text':
                    for key in STREAM_FIELDS:
                        if isinstance(record.get(key), str):
                            legacy.setdefault(key, record[key])
                    if len(legacy) == len(STREAM_FIELDS):
                        state = legacy
                        break
            elif encoding == 2:
                if legacy:
                    pending.append((json.dumps({'kind': 'text', **legacy}), 1))
                    legacy = {}
                if 'state' in record:
                    state = dict(record['state'])
                    break
                pending.append((body, encoding))
            else:
                raise sqlite3.DatabaseError('Unsupported stored event encoding')
        else:
            state = legacy
        for body, encoding in reversed(pending):
            self._decode(body, encoding, state)
        return state

    def _remember_replay(self, jid, seq, state):
        key = (jid, seq)
        self._replay_cache[key] = dict(state)
        self._replay_cache.move_to_end(key)
        # Unicode worst case is four bytes/codepoint. Keep this auxiliary cache
        # below approximately 8 MiB and 32 cursors, independently of job count.
        while self._replay_cache and (len(self._replay_cache) > 32 or
                sum(len(value) for cached in self._replay_cache.values() for value in cached.values()) > 2 * 1024 * 1024):
            self._replay_cache.popitem(last=False)

    @storage_operation
    def event(self, jid, kind, *, job_update=None, **data):
        if job_update is not None and (job_update['id'] != jid or kind not in STREAM_JOB_EVENTS):
            raise ValueError('Invalid streaming job update')
        item = {'kind': kind, 'time': time.time(), **data}
        with self.lock:
            state = None
            checkpoint_size = 0
            encoding, checkpoint, body = 1, 0, item
            if kind == 'text':
                head = self._write_heads.get(jid)
                if head is None:
                    # Reconstruct only this job's prefix on a cold writer. This
                    # also keeps deltas compact if active jobs exceed the cache.
                    latest = self.db.execute('SELECT COALESCE(MAX(seq),0) FROM events WHERE job=?', (jid,)).fetchone()[0]
                    previous = self._state_at(jid, latest)
                    checkpoint_size = max(2048, sum(len(v) for v in previous.values()))
                else:
                    previous, checkpoint_size, _ = head
                state = dict(previous)
                fields = {}
                rest = dict(item)
                for key in STREAM_FIELDS:
                    if not isinstance(item.get(key), str):
                        continue
                    value = rest.pop(key)
                    old = previous.get(key, '')
                    keep = min(len(old), len(value))
                    if not value.startswith(old):
                        keep = 0
                        for left, right in zip(old, value):
                            if left != right:
                                break
                            keep += 1
                    fields[key] = [keep, value[keep:]]
                    state[key] = value
                size = sum(len(v) for v in state.values())
                # Geometric checkpoints keep full-text storage linear. Fixed
                # every-N-event checkpoints would still grow quadratically.
                checkpoint = int(not previous or size >= checkpoint_size * 2)
                if checkpoint:
                    checkpoint_size = max(2048, size)
                    body = {'item': rest, 'fields': list(fields), 'state': state}
                else:
                    body = {'item': rest, 'fields': fields}
                encoding = 2
            with self.db:
                seq = self.db.execute('INSERT INTO events(job,body,encoding,checkpoint,kind) VALUES(?,?,?,?,?)',
                                      (jid, json.dumps(body), encoding, checkpoint, kind)).lastrowid
                if job_update is not None:
                    # Keep metadata on separate, small pages. Updating a column
                    # in jobs can still rewrite its large JSON overflow pages.
                    # The event and its latest-state pointer commit atomically;
                    # no timer or in-memory buffer can lose acknowledged text.
                    update = {key: job_update[key] for key in STREAM_JOB_FIELDS if key in job_update}
                    self.db.execute('''INSERT INTO job_streams(job,body,text_seq) VALUES(?,?,?)
                        ON CONFLICT(job) DO UPDATE SET body=excluded.body,
                        text_seq=COALESCE(excluded.text_seq,job_streams.text_seq)''',
                        (jid, json.dumps(update), seq if kind == 'text' else None))
            # Only cache a prefix after its corresponding transaction commits.
            if state is not None:
                self._write_heads[jid] = (state, checkpoint_size, seq)
                self._write_heads.move_to_end(jid)
                while len(self._write_heads) > 32:
                    self._write_heads.popitem(last=False)
            if kind == 'finished':
                self._write_heads.pop(jid, None)

    @storage_operation
    def events(self, jid, after=0):
        with self.lock:
            rows = self.db.execute('SELECT seq,body,encoding FROM events WHERE job=? AND seq>? ORDER BY seq LIMIT 300',
                                   (jid, after)).fetchall()
            if not rows:
                return []
            cached = self._replay_cache.get((jid, after))
            if cached is not None:
                state = dict(cached)
            else:
                # Legacy cumulative records and a first compact checkpoint are
                # self-contained. Only a delta needs the preceding prefix.
                first_compact = next((json.loads(body) for _, body, encoding in rows if encoding == 2), None)
                state = self._state_at(jid, after) if first_compact is not None and 'state' not in first_compact else {}
            result = [{'seq': seq, **self._decode(body, encoding, state)} for seq, body, encoding in rows]
            self._remember_replay(jid, rows[-1][0], state)
            return result

    @storage_operation
    def compact_events(self, before, limit=20):
        """Opt-in replay compaction for old terminal jobs; never deletes results.

        Collapse repeated text events to their last cumulative checkpoint. Keep
        every non-text event, final text sequence ID, job, retry key and file.
        Old stream cursors remain usable but skip compacted intermediate text.
        No automatic retention policy calls this method by default.
        """
        limit = max(1, min(int(limit), 100))
        with self.lock:
            jobs = self.db.execute(f"""SELECT j.id FROM jobs j JOIN events e ON e.job=j.id AND {text_events('e')}
                WHERE j.status IN ('completed','failed','cancelled','interrupted') AND j.updated<?
                GROUP BY j.id HAVING COUNT(*)>1 ORDER BY j.updated LIMIT ?""", (float(before), limit)).fetchall()
            removed = 0
            for (jid,) in jobs:
                row = self.db.execute(f"SELECT seq,body,encoding FROM events WHERE job=? AND {text_events()} ORDER BY seq DESC LIMIT 1", (jid,)).fetchone()
                state = self._state_at(jid, row[0] - 1)
                item = self._decode(row[1], row[2], state)
                fields = [key for key in STREAM_FIELDS if isinstance(item.get(key), str)]
                payload = {'item': {key: value for key, value in item.items() if key not in fields},
                           'fields': fields, 'state': state}
                with self.db:
                    self.db.execute("UPDATE events SET body=?,encoding=2,checkpoint=1,kind='text' WHERE seq=?", (json.dumps(payload), row[0]))
                    removed += self.db.execute(f"DELETE FROM events WHERE job=? AND {text_events()} AND seq<?", (jid, row[0])).rowcount
                self._write_heads.pop(jid, None)
                for key in [key for key in self._replay_cache if key[0] == jid]:
                    del self._replay_cache[key]
            return {'jobs': len(jobs), 'text_events_removed': removed}

    def close(self):
        with self.lock:
            if self._closed:
                return
            self._closed = True
            try:
                if self.db is not None:
                    self.db.close()
            finally:
                self._owner.close()

    def __del__(self):
        # Explicit close remains the normal lifecycle; failed initialization or
        # a discarded fixture must not keep an OS lock alive indefinitely.
        try:
            if hasattr(self, '_owner'):
                self.close()
        except Exception:
            pass
