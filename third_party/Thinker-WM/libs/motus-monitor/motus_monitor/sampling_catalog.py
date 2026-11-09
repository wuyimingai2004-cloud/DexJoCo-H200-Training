"""Export lightweight task labels on the training host (rank zero only)."""
import json
import sqlite3
from pathlib import Path
from .store import atomic_write_json


def export_sampling_catalog(run_dir, config):
    episodes, task_labels = [], []
    for source in (config or {}).get('dataset', {}).get('sources', []):
        cat = source.get('catalog_path')
        ranges = source.get('pool', {}).get('path')
        if not cat or not ranges or not Path(cat).is_file() or not Path(ranges).is_file():
            continue
        labels = {}
        with Path(cat).open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                item = json.loads(line)
                labels[item['episode_id']] = item.get('meta', {}).get('task_name', '未知任务')
        with sqlite3.connect(Path(ranges).resolve().as_uri() + '?mode=ro', uri=True) as db:
            rows = db.execute('SELECT episode_id,intervals FROM episode_ranges').fetchall()
        for eid, intervals in rows:
            episodes.append(dict(source=source['name'], episode_id=eid,
                task=labels.get(eid, '未知任务'),
                total_windows=sum(b-a+1 for a,b in json.loads(intervals))))
        task_labels.extend(dict(source=source['name'], episode_id=eid, task=task)
                           for eid, task in labels.items())
    if episodes:
        atomic_write_json(Path(run_dir)/'sampling_catalog.json',
                          dict(episodes=episodes, task_labels=task_labels))
