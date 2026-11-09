"""Globally dealt 2x4 windows, with seed/phase hierarchy for expert recovery."""
import json
from collections import defaultdict
from pathlib import Path
from thinker_wm.data_prepare.global_cycled_ranges import GlobalCycledRangesPool, create_training_data_pipeline, install_exact_resume
from thinker_wm.data_prepare.sampling_ranges import _feistel_permute
from thinker_wm.data_prepare.pool_providers import register_pool_builder

class SeedCycledRangesPool(GlobalCycledRangesPool):
    def __init__(self,*args,hierarchy_path=None,**kwargs):
        super().__init__(*args,**kwargs)
        self._hierarchy=None
        if hierarchy_path:
            records={r['episode_id']:r for r in map(json.loads,Path(hierarchy_path).read_text().splitlines())}
            tasks=[]
            for eps in self._task_episodes:
                buckets={}
                for ep in eps:
                    r=records[ep.episode_id]
                    bucket='failure' if r['policy_success'] is False else 'success'
                    buckets.setdefault(bucket,{}).setdefault(int(r['seed']),{}).setdefault(r['phase'] or 'unknown',[]).append(ep)
                tasks.append(buckets)
            self._hierarchy=tasks

    def deal_at(self,ordinal):
        if self._hierarchy is None:return super().deal_at(ordinal)
        turn,seat=divmod(int(ordinal),len(self._tasks))
        task=_feistel_permute(seat,len(self._tasks),self._key('task',turn))
        buckets=self._hierarchy[task]
        if len(buckets)==2:
            # Give each failed-rollout seed twice a successful seed's weight.
            # A fixed 2:1 bucket ratio would overfit tasks with very few failures.
            failure_weight=2*len(buckets['failure'])
            total=failure_weight+len(buckets['success'])
            before=turn*failure_weight//total
            after=(turn+1)*failure_weight//total
            bucket='failure' if after>before else 'success'
            draw=after-1 if bucket=='failure' else turn-before
        else:bucket=next(iter(buckets));draw=turn
        seeds=sorted(buckets[bucket]);visit,seat=divmod(draw,len(seeds))
        seed=seeds[_feistel_permute(seat,len(seeds),self._key('seed',task,bucket,0))]
        phases=sorted(buckets[bucket][seed]);phase_visit,seat=divmod(visit,len(phases))
        phase=phases[_feistel_permute(seat,len(phases),self._key('phase',task,seed,0))]
        eps=sorted(buckets[bucket][seed][phase],key=lambda ep:ep.episode_id)
        use,seat=divmod(phase_visit,len(eps))
        ep=eps[_feistel_permute(seat,len(eps),self._key('segment',task,seed,phase,0))]
        return task,ep,use

def _build(repository,*,path,**options):
    return SeedCycledRangesPool(path,repository,**options)

register_pool_builder('seed_cycled_ranges',_build,known_options=(
    'path','num_video_frames','stride_max','text_choice','action_start_offset',
    'group_size','cycle_base_step','source_name','source_names','source_weights',
    'world_size','schedule_steps','episodes_per_batch','hierarchy_path'))
