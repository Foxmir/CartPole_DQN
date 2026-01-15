# CartPole_DQN/src/agents/template_agent.py
# Agent script template; not executed in practice (may be extended into a base class later).
# It may contain syntax issues; for reference only.


from ..utils.massage import print_message

class BaseAgent:
    """Base class for reinforcement learning agents (parent class)."""

    # !!! Note: removed default values for agent_name and learning_rate !!!
    def __init__(self, agent_name, learning_rate):
        """
        Initialize the base agent.
        Args:
            agent_name (str): Agent name (required).
            learning_rate (float): Learning rate (required).
        """
        self.agent_name = agent_name
        # !!! Note: directly use the passed-in learning_rate parameter !!!
        self.lr = learning_rate

        # Use a helper to print initialization info
        print_message(f"Agent '{self.agent_name}' initialized with learning rate: {self.lr}")

    def select_action(self, state):
        """
        Select an action based on the current state (dummy implementation).
        Args:
            state: Current environment state.
        Returns:
            An action (returns a dummy action 0).
        """
        print_message(f"Agent '{self.agent_name}' is selecting an action for state: {state}")
        # In a real agent, this would involve a policy network, exploration (e.g., epsilon-greedy), etc.
        return 0 # Return a dummy action

    def learn(self, experiences):
        """
        Learn from a batch of experiences (dummy implementation).
        Args:
            experiences (list): Experience batch, typically tuples of (state, action, reward, next_state, done).
        """
        print_message(f"Agent '{self.agent_name}' is learning from {len(experiences)} experiences.")
        # In a real agent, this would run gradient descent and update network weights
        pass # pass means do nothing for now