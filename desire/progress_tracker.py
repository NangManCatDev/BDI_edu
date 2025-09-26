class ProgressTracker:
    def __init__(self):
        self.tracking = {}

    def start_tracking(self, goal):
        self.tracking = {"goal": goal, "progress": 0}
        return self.tracking

    def update_progress(self, step):
        self.tracking["progress"] += step
        return self.tracking
