import random
from loguru import logger
from src.phase0_probe.difficulty_mapper import DifficultyMapper
from .accuracy_tracker import AccuracyTracker

# Hard ceiling – never go above this
MAX_DIFFICULTY_CEILING = 0.85
# Floor cap – never push floor above this
MAX_DIFFICULTY_FLOOR = 0.70

class CurriculumEngine:
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
                logger.warning("Difficulty map not found. Using empty fallback!")
        self.min_difficulty = initial_range[0]
        self.max_difficulty = min(initial_range[1], MAX_DIFFICULTY_CEILING)
        self.tracker = AccuracyTracker(window_size=window_size, stability_threshold=stability_threshold)
        logger.info(f"CurriculumEngine: range [{self.min_difficulty}, {self.max_difficulty}] | ceiling={MAX_DIFFICULTY_CEILING}")

    @property
    def current_range(self):
        return [self.min_difficulty, self.max_difficulty]

    @current_range.setter
    def current_range(self, val):
        self.min_difficulty = val[0]
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
            logger.warning("Active pool empty → fallback to full map")
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
        self.min_difficulty = max(0.0, round(self.min_difficulty - 0.1, 2))
        self.max_difficulty = min(MAX_DIFFICULTY_CEILING, round(self.max_difficulty + 0.1, 2))
        logger.info(f"Temp expansion: [{self.min_difficulty}, {self.max_difficulty}]")

    def collapse_temporarily(self):
        """Step back difficulty bounds if model is failing."""
        self.max_difficulty = max(0.4, round(self.max_difficulty - 0.1, 2))
        self.min_difficulty = max(0.1, round(self.min_difficulty - 0.1, 2))
        logger.warning(f"Collapsed range: [{self.min_difficulty}, {self.max_difficulty}]")

    def update_accuracy(self, *args, **kwargs):
        is_correct = kwargs.get("is_correct", None)
        if is_correct is None and len(args) >= 2:
            is_correct = args[1]
        if is_correct is None:
            return
        self.tracker.add(is_correct)
        if self.tracker.is_stable():
            old_min, old_max = self.min_difficulty, self.max_difficulty
            new_max = min(MAX_DIFFICULTY_CEILING, round(self.max_difficulty + 0.1, 2))
            new_min = min(MAX_DIFFICULTY_FLOOR, round(self.min_difficulty + 0.05, 2))
            if new_max == old_max and new_min == old_min:
                logger.info(f"Already at max ceiling [{old_min},{old_max}]. No expansion.")
            else:
                self.max_difficulty = new_max
                self.min_difficulty = new_min
                logger.info(f"Expanded: [{old_min:.2f},{old_max:.2f}] → [{new_min:.2f},{new_max:.2f}]")
            self.tracker.reset()