import random
from loguru import logger
from src.phase0_probe.difficulty_mapper import DifficultyMapper
from .accuracy_tracker import AccuracyTracker

# ─── DIFFICULTY CEILING — DO NOT CHANGE THIS ───────────────────────────────
# The maximum allowed difficulty ceiling during RL training.
# 0.85 means: at most, the model will face questions it solved 15% of the time
# during Phase 0 probing. Going above this causes zero-success batches which
# produce zero reward variance and crash training stability (as seen at step 220).
MAX_DIFFICULTY_CEILING = 0.85

# The maximum allowed difficulty floor.
# 0.70 means: the model will always see at least some questions it was getting
# right, maintaining enough reward variance to learn from.
MAX_DIFFICULTY_FLOOR = 0.70


class CurriculumEngine:
    """
    Adaptive curriculum engine.
    Samples training questions based on model capabilities, filtering
    for the optimal 'learning zone' (40-70% difficulty by default).
    Dynamically expands the difficulty bounds as training accuracy stabilizes.
    
    KEY CHANGE v3: Hard ceiling at MAX_DIFFICULTY_CEILING (0.85) to prevent
    the collapse seen at step 220 where difficulty reached [0.55, 1.0].
    """
    def __init__(
        self,
        difficulty_map_path="difficulty_map.json",
        initial_range=(0.4, 0.7),
        window_size=50,
        stability_threshold=0.7,
        difficulty_mapper=None,
    ):
        if difficulty_mapper is not None:
            self.mapper = difficulty_mapper
        else:
            self.mapper = DifficultyMapper()
            try:
                self.mapper.load_map(difficulty_map_path)
            except FileNotFoundError:
                logger.warning(
                    f"Difficulty map not found at {difficulty_map_path}. "
                    "Initializing with empty/mock fallback!"
                )

        self.min_difficulty = initial_range[0]
        self.max_difficulty = min(initial_range[1], MAX_DIFFICULTY_CEILING)

        self.tracker = AccuracyTracker(
            window_size=window_size,
            stability_threshold=stability_threshold,
        )
        logger.info(
            f"CurriculumEngine initialized with difficulty range "
            f"[{self.min_difficulty}, {self.max_difficulty}]. "
            f"Hard ceiling: {MAX_DIFFICULTY_CEILING}"
        )

    @property
    def current_range(self):
        return [self.min_difficulty, self.max_difficulty]

    @current_range.setter
    def current_range(self, val):
        self.min_difficulty = val[0]
        # ENFORCE CEILING on setter too — resuming from a checkpoint with bad values
        self.max_difficulty = min(val[1], MAX_DIFFICULTY_CEILING)

    def get_active_pool(self) -> list:
        if not self.mapper.difficulty_map:
            return []
        pool = []
        for q_id, info in self.mapper.difficulty_map.items():
            diff = info.get("difficulty", 0.5)
            if self.min_difficulty <= diff <= self.max_difficulty:
                pool.append({"id": q_id, **info})
        return pool

    def sample_question(self) -> dict:
        pool = self.get_active_pool()
        if not pool:
            logger.warning(
                "Active curriculum pool is empty. Fallback sampling from complete map!"
            )
            if self.mapper.difficulty_map:
                q_id = random.choice(list(self.mapper.difficulty_map.keys()))
                return {"id": q_id, **self.mapper.difficulty_map[q_id]}
            return None
        return random.choice(pool)

    def sample_batch(self, batch_size=8) -> list:
        pool = self.get_active_pool()
        if not pool:
            if self.mapper.difficulty_map:
                all_keys = list(self.mapper.difficulty_map.keys())
                samples = random.sample(all_keys, min(batch_size, len(all_keys)))
                return [{"id": k, **self.mapper.difficulty_map[k]} for k in samples]
            return []
        if batch_size <= len(pool):
            return random.sample(pool, batch_size)
        return [random.choice(pool) for _ in range(batch_size)]

    def get_next_batch(self, batch_size=1) -> list:
        return self.sample_batch(batch_size=batch_size)

    def expand_range_temporarily(self):
        """Temporarily expands bounds when active pool is dry. Ceiling enforced."""
        self.min_difficulty = max(0.0, round(self.min_difficulty - 0.1, 2))
        # CEILING ENFORCED HERE
        self.max_difficulty = min(
            MAX_DIFFICULTY_CEILING,
            round(self.max_difficulty + 0.1, 2)
        )
        logger.info(
            f"Curriculum bounds temporarily expanded to: "
            f"[{self.min_difficulty}, {self.max_difficulty}]"
        )

    def collapse_temporarily(self):
        """Steps back the bounds if the model collapses."""
        self.max_difficulty = max(0.4, round(self.max_difficulty - 0.1, 2))
        self.min_difficulty = max(0.1, round(self.min_difficulty - 0.1, 2))
        logger.warning(
            f"Curriculum bounds collapsed (safety fallback) to: "
            f"[{self.min_difficulty}, {self.max_difficulty}]"
        )

    def update_accuracy(self, *args, **kwargs):
        """
        Registers correctness and updates tracker.
        Expands difficulty range if accuracy has stabilized.
        Ceiling is enforced on all expansions.
        """
        is_correct = kwargs.get("is_correct", None)
        if is_correct is None:
            if len(args) == 1:
                is_correct = args[0]
            elif len(args) == 2:
                is_correct = args[1]

        self.tracker.add(is_correct)

        if self.tracker.is_stable():
            old_max = self.max_difficulty
            old_min = self.min_difficulty

            # CEILING ENFORCED — never go above MAX_DIFFICULTY_CEILING
            new_max = min(
                MAX_DIFFICULTY_CEILING,
                round(self.max_difficulty + 0.1, 2)
            )
            # FLOOR CAP — never push the floor above MAX_DIFFICULTY_FLOOR
            new_min = min(
                MAX_DIFFICULTY_FLOOR,
                round(self.min_difficulty + 0.05, 2)
            )

            if new_max == old_max and new_min == old_min:
                # Already at ceiling — nothing to expand
                logger.info(
                    f"Curriculum already at max allowed range "
                    f"[{old_min}, {old_max}]. Ceiling={MAX_DIFFICULTY_CEILING}. "
                    "No expansion. Tracker reset."
                )
            else:
                self.max_difficulty = new_max
                self.min_difficulty = new_min
                logger.info(
                    f"Curriculum expanded: [{old_min:.2f}, {old_max:.2f}] → "
                    f"[{new_min:.2f}, {new_max:.2f}]"
                )

            self.tracker.reset()
