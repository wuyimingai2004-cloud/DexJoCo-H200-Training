"""Two trajectories per batch, with complementary strata staggered by episode."""
from thinker_wm.data_prepare.global_cycled_ranges import (
    GlobalCycledRangesPool, create_training_data_pipeline, install_exact_resume,
)
from thinker_wm.data_prepare.pool_providers import register_pool_builder
from thinker_wm.data_prepare.sampling_ranges import _feistel_permute


class PhasedGlobalRangesPool(GlobalCycledRangesPool):
    def initial_parity(self, episode):
        return self._key("initial_parity", episode.episode_id, 0) & 1

    def _pick_start(self, episode, draw, rng):
        use, slot = divmod(int(draw), self.group_size)
        strata = 2 * self.group_size
        parity = (self.initial_parity(episode) + use) % 2
        stratum = 2 * slot + parity
        lo = stratum * episode.total_starts // strata
        hi = (stratum + 1) * episode.total_starts // strata
        if hi <= lo:
            return episode.start_at(min(lo, episode.total_starts - 1))
        rotation, position = divmod(use // 2, hi - lo)
        key = self._key("window", episode.episode_id, stratum, rotation)
        return episode.start_at(lo + _feistel_permute(position, hi - lo, key))


def _build(repository, *, path, **options):
    return PhasedGlobalRangesPool(path, repository, **options)


register_pool_builder("phased_global_ranges", _build, known_options=(
    "path", "num_video_frames", "stride_max", "text_choice", "action_start_offset",
    "group_size", "cycle_base_step", "source_name", "source_names", "source_weights",
    "world_size", "schedule_steps", "episodes_per_batch",
))
