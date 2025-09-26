class StateManager:
    def __init__(self):
        self.state = {"student_level": "beginner", "progress": 0}

    def load_initial_state(self):
        return self.state

    def update_state(self, key, value):
        self.state[key] = value
        return self.state
