class Agent:
    def save_models(self, path):
        pass
    def load_models(self, path):
        pass
    def select_action(self, state):
        pass
    def select_exploratory_action(self, state):
        pass
    def train(self, state, action, next_state, reward, done):
        pass
    def get_signature_values(self, agent_type_name, train_step_count, train_seed):
        signature =  [agent_type_name, train_step_count, train_seed]
        return signature
