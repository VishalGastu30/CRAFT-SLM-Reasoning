import random
from loguru import logger
from src.phase0_probe.difficulty_mapper import DifficultyMapper
from .accuracy_tracker import AccuracyTracker

class CurriculumEngine:
    """
    Adaptive curriculum engine.
    Samples training questions based on model capabilities, filtering
    for the optimal 'learning zone' (40-70% difficulty by default).
    Dynamically expands the difficulty bounds as training accuracy stabilizes.
    """
    def __init__(self, difficulty_map_path="difficulty_map.json", initial_range=(0.4, 0.7), window_size=50, stability_threshold=0.7, difficulty_mapper=None):
        if difficulty_mapper is not None:
            self.mapper = difficulty_mapper
        else:
            self.mapper = DifficultyMapper()
            try:
                self.mapper.load_map(difficulty_map_path)
            except FileNotFoundError:
                logger.warning(f"Difficulty map not found at {difficulty_map_path}. Initializing with empty/mock fallback!")
            
        self.min_difficulty = initial_range[0]
        self.max_difficulty = initial_range[1]
        
        self.tracker = AccuracyTracker(window_size=window_size, stability_threshold=stability_threshold)
        logger.info(f"CurriculumEngine initialized with difficulty range [{self.min_difficulty}, {self.max_difficulty}]")

    @property
    def current_range(self):
        """Dynamic getter for compatibility with new loop's current_range usage."""
        return [self.min_difficulty, self.max_difficulty]

    @current_range.setter
    def current_range(self, val):
        """Dynamic setter for compatibility with new loop's current_range usage."""
        self.min_difficulty = val[0]
        self.max_difficulty = val[1]

    def get_active_pool(self) -> list:
        """
        Filters all questions from the difficulty map that fall within 
        the current difficulty bounds.
        """
        if not self.mapper.difficulty_map:
            return []
            
        pool = []
        for q_id, info in self.mapper.difficulty_map.items():
            diff = info.get("difficulty", 0.5)
            if self.min_difficulty <= diff <= self.max_difficulty:
                pool.append({
                    "id": q_id,
                    **info
                })
        return pool

    def sample_question(self) -> dict:
        """
        Samples a single question from the active difficulty pool.
        Falls back to a random sample or None if no pool is available.
        """
        pool = self.get_active_pool()
        if not pool:
            logger.warning("Active curriculum pool is empty. Fallback sampling from complete map!")
            if self.mapper.difficulty_map:
                q_id = random.choice(list(self.mapper.difficulty_map.keys()))
                return {"id": q_id, **self.mapper.difficulty_map[q_id]}
            return None
            
        return random.choice(pool)

    def sample_batch(self, batch_size=8) -> list:
        """Samples a batch of questions from the active pool."""
        pool = self.get_active_pool()
        if not pool:
            if self.mapper.difficulty_map:
                all_keys = list(self.mapper.difficulty_map.keys())
                samples = random.sample(all_keys, min(batch_size, len(all_keys)))
                return [{"id": k, **self.mapper.difficulty_map[k]} for k in samples]
            return []
            
        if batch_size <= len(pool):
            return random.sample(pool, batch_size)
        else:
            return [random.choice(pool) for _ in range(batch_size)]

    def get_next_batch(self, batch_size=1) -> list:
        """Compatibility method mapping to sample_batch."""
        return self.sample_batch(batch_size=batch_size)

    def expand_range_temporarily(self):
        """Temporarily expands the difficulty bounds when active pool is dry."""
        self.min_difficulty = max(0.0, round(self.min_difficulty - 0.1, 2))
        self.max_difficulty = min(1.0, round(self.max_difficulty + 0.1, 2))
        logger.info(f"Curriculum bounds temporarily expanded to: [{self.min_difficulty}, {self.max_difficulty}]")

    def update_accuracy(self, *args, **kwargs):
        """
        Registers model correctness and updates the accuracy tracker.
        Accepts:
            - update_accuracy(is_correct)
            - update_accuracy(question_id, is_correct)
            - update_accuracy(question_id=..., is_correct=...)
        If accuracy stabilizes, expands difficulty range to include harder questions.
        """
        is_correct = kwargs.get("is_correct", None)
        if is_correct is None:
            if len(args) == 1:
                is_correct = args[0]
            elif len(args) == 2:
                # args are (question_id, is_correct)
                is_correct = args[1]
        
        self.tracker.add(is_correct)
        
        # Check stability and shift difficulty
        if self.tracker.is_stable():
            old_max = self.max_difficulty
            self.max_difficulty = min(1.0, round(self.max_difficulty + 0.1, 2))
            self.min_difficulty = min(0.8, round(self.min_difficulty + 0.05, 2))
            logger.info(f"Curriculum expanded! Learning zone difficulty shifted: [{old_max}] -> [{self.max_difficulty}]")
            self.tracker.reset()
