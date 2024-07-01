from symaware.base import Agent
class TasMasAgent(Agent):
    def step(self):
        # Your implementation here
        # Example:
        # Compute all the components in the order they were added
        for component in self.components:
            component.compute_and_update()