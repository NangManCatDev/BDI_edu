class Planner:
    def create_plan(self, goal, state):
        return {
            "goal": goal,
            "steps": ["review basics", "teach new concept", "give quiz"],
            "state": state,
        }
