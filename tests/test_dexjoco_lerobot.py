"""LeRobot episode boundaries, camera ordering and existing Motus pipeline."""
from copy import deepcopy
import json
import hashlib
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from thinker_wm.data_prepare.dexjoco.contracts import TASKS, validate_manifest
from thinker_wm.data_prepare.dexjoco.lerobot import LeRobotV2
from thinker_wm.data_prepare.dexjoco.prepare import prepare_release
from thinker_wm.data_prepare.dexjoco.download import download_all


def fixture(root, task, variant='rand_obj', count=31, action_offset=100.):
    (root / 'meta').mkdir(parents=True)
    camera = 'random_camera' if variant == 'rand_full' else ('ego' if task.arms == 2 else 'ego_right' if task.name == 'click_mouse' else 'front')
    cameras = [camera, 'wrist'] if task.arms == 1 else [camera, 'wrist_left', 'wrist_right']
    info = dict(codebase_version='v2.0', fps=30, chunks_size=1000, total_episodes=2,
                data_path='data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
                video_path='videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4',
                features={'action': dict(shape=[task.dimension]), 'observation.state': dict(shape=[23 * task.arms]),
                          **{'observation.images.' + k: dict(dtype='video') for k in cameras}})
    (root / 'meta/info.json').write_text(json.dumps(info))
    (root / 'meta/episodes.jsonl').write_text('\n'.join(json.dumps(dict(episode_index=i, length=count, tasks=['fixture task'])) for i in range(2)))
    rows = []
    for i in range(2):
        state = np.zeros((count, task.arms * 23), dtype=np.float32)
        state[:, 0], state[:, 3] = i + 1, 1
        if task.arms == 2:
            state[:, 7], state[:, 10] = i + 2, 1
            state[:, 14:30], state[:, 30:46] = .2, .4
        action = np.zeros((count, task.dimension), dtype=np.float32)
        action[:, 0] = i * action_offset + np.arange(count) * .01
        table = pa.table({'observation.state': state.tolist(), 'action': action.tolist(),
                          'timestamp': (np.arange(count) / 30).astype(np.float32), 'frame_index': np.arange(count),
                          'episode_index': np.full(count, i)})
        path = root / info['data_path'].format(episode_chunk=0, episode_index=i)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        for camera in cameras:
            p = root / info['video_path'].format(episode_chunk=0, episode_index=i, video_key='observation.images.'+camera)
            p.parent.mkdir(parents=True, exist_ok=True)
            w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*'mp4v'), 30, (16,16))
            if not w.isOpened():
                raise RuntimeError('fixture encoder unavailable')
            for t in range(count):
                w.write(np.full((16,16,3), 30 + t * 3 + i * 20, dtype=np.uint8))
            w.release()
        rows.append(dict(task=task.name, episode_dir=str(root), format='lerobot_v2', dataset_variant=variant,
                         episode_index=i, source_group=f'original_{i}', layout_id=f'layout_{i}',
                         split='train' if i == 0 else 'dev', instruction='fixture task'))
    return dict(schema='dexjoco-training-manifest/v1', alignment='obs_t_action_t', episodes=rows)


class LeRobotTest(unittest.TestCase):
    def test_six_tasks_train_statistics_and_video_reader(self):
        from thinker_wm.data_prepare.dexjoco.reader import DexJoCoReader
        from motus_data.repository import EpisodeRepository
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            for task in TASKS:
                rows.extend(fixture(root / task.name, task)['episodes'])
                dataset = LeRobotV2(root / task.name, task, 'rand_obj')
                s, a, _, videos = dataset.read(0, 'fixture task')
                self.assertEqual(a.shape, (31, task.dimension))
                if task.arms == 2:
                    np.testing.assert_allclose(s[:, 6:22], .2)
                    np.testing.assert_allclose(s[:, 28:44], .4)
                    self.assertIn('wrist_right', str(videos[1]))
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps(dict(schema='dexjoco-training-manifest/v1', alignment='obs_t_action_t', episodes=rows)))
            output = root / 'release'
            result = prepare_release(manifest, output, [t.name for t in TASKS], target_episodes=1)
            self.assertEqual(len(result['tasks']), 6)
            for task in TASKS:
                stats = json.loads((output / task.name / 'action_stats.json').read_text())
                self.assertAlmostEqual(stats['mean'][0], .15, places=5)
                repo = EpisodeRepository(catalog_path=output / task.name / 'train.episode_catalog.jsonl',
                                         readers={'dexjoco': DexJoCoReader()})
                try:
                    episode = repo.get_episode(next(repo.iter_episode_ids()))
                    frames = episode.read_video([0, 2, 24], output_format='rgb_uint8_hwc')
                    self.assertEqual(tuple(frames.shape), (3,480,640,3))
                    self.assertGreater(float(frames[2].float().mean()), float(frames[0].float().mean()) + 15)
                finally:
                    repo.close()

    def test_timestamp_order_instruction_variant_and_version_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            fixture(root, TASKS[0])
            dataset=LeRobotV2(root,TASKS[0],'rand_obj')
            with self.assertRaisesRegex(ValueError,'instruction'):
                dataset.read(0,'wrong task')
            with self.assertRaisesRegex(ValueError,'missing'):
                LeRobotV2(root,TASKS[0],'rand_full')
            p,_=dataset.paths(0)
            table=pq.read_table(p)
            values=table.to_pydict()
            values['frame_index'][3]=9
            pq.write_table(pa.table(values),p)
            with self.assertRaisesRegex(ValueError,'contiguous'):
                dataset.read(0,'fixture task')
            values=table.to_pydict()
            values['timestamp'][3]=.9
            pq.write_table(pa.table(values),p)
            with self.assertRaisesRegex(ValueError,'30 Hz'):
                dataset.read(0,'fixture task')
            values=table.to_pydict()
            values['episode_index'][3]=1
            pq.write_table(pa.table(values),p)
            with self.assertRaisesRegex(ValueError,'different episode'):
                dataset.read(0,'fixture task')
            info=json.loads((root/'meta/info.json').read_text()); info['codebase_version']='v3.0'
            (root/'meta/info.json').write_text(json.dumps(info))
            with self.assertRaisesRegex(ValueError,'v3'):
                LeRobotV2(root,TASKS[0],'rand_obj')

    def test_same_root_distinct_episodes_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            d=fixture(Path(temp),TASKS[0]); validate_manifest(d)
            dup=deepcopy(d); dup['episodes'][1]['episode_index']=0
            with self.assertRaisesRegex(ValueError,'duplicate'):
                validate_manifest(dup)
            mix=deepcopy(d); mix['episodes'][1]['dataset_variant']='rand_full'
            with self.assertRaisesRegex(ValueError,'mix'):
                validate_manifest(mix)

    def test_paths_cannot_escape_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);fixture(root,TASKS[0])
            ds=LeRobotV2(root,TASKS[0],'rand_obj');ds.info['data_path']='../outside.parquet'
            with self.assertRaisesRegex(ValueError,'escapes'):
                ds.paths(0)

    def test_download_resumes_and_checks_hash_before_renaming(self):
        payload = b'archive fixture'
        manifest = dict(repo_id='example/dataset', revision='fixed', tasks={
            'singlearm/pick_bucket': dict(zip_bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())})
        with tempfile.TemporaryDirectory() as temp, patch(
                'thinker_wm.data_prepare.dexjoco.download.json.loads', return_value=manifest):
            root = Path(temp)
            part = root / 'pick_bucket.zip.part'
            part.write_bytes(payload[:4])
            def transfer(command, **kwargs):
                self.assertIn('--continue-at', command)
                self.assertEqual(part.read_bytes(), payload[:4])
                part.write_bytes(payload)
            with patch('subprocess.run', side_effect=transfer) as run:
                download_all(root)
                run.assert_called_once()
            self.assertFalse(part.exists())
            self.assertEqual((root / 'pick_bucket.zip').read_bytes(), payload)
            with patch('subprocess.run') as run:
                download_all(root)
                run.assert_not_called()

    def test_download_failure_retains_partial_and_never_marks_complete(self):
        manifest = dict(repo_id='example/dataset', revision='fixed', tasks={
            'singlearm/pick_bucket': dict(zip_bytes=3, sha256=hashlib.sha256(b'abc').hexdigest())})
        with tempfile.TemporaryDirectory() as temp, patch(
                'thinker_wm.data_prepare.dexjoco.download.json.loads', return_value=manifest):
            root=Path(temp); part=root/'pick_bucket.zip.part';part.write_bytes(b'a')
            with patch('subprocess.run', side_effect=subprocess.CalledProcessError(18, 'curl')):
                with self.assertRaises(subprocess.CalledProcessError):
                    download_all(root)
            self.assertEqual(part.read_bytes(),b'a')
            self.assertFalse((root/'download_manifest.json').exists())
            with patch('subprocess.run', side_effect=lambda *args, **kwargs: part.write_bytes(b'bad')):
                with self.assertRaisesRegex(RuntimeError,'SHA256'):
                    download_all(root)
            self.assertFalse((root/'pick_bucket.zip').exists())

    def test_download_verify_only_refuses_missing_without_starting_transfer(self):
        with tempfile.TemporaryDirectory() as temp, patch('subprocess.run') as run:
            with self.assertRaisesRegex(RuntimeError,'missing/corrupt'):
                download_all(Path(temp),only=['pick_bucket'],verify_only=True)
            run.assert_not_called()
            with self.assertRaisesRegex(ValueError,'unknown'):
                download_all(Path(temp),only=['not_a_task'])


if __name__ == '__main__':
    unittest.main()
