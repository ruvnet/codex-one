class InputField:
    pass

class OutputField:
    pass

class Signature(dict):
    pass

class Prediction:
    def __init__(self, **fields):
        for k, v in fields.items():
            setattr(self, k, v)

class Module:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    def forward(self, *args, **kwargs):
        raise NotImplementedError

class Tunable:
    def __init__(self, name: str, init_val: float = 0.5):
        self.name = name
        self.value = init_val
    def __call__(self):
        return self.value
    def set(self, val: float):
        self.value = val
class SimpleGRPO:
    def __init__(self, module, epochs: int = 1, lr: float = 0.1):
        self.module = module
        self.epochs = epochs
        self.lr = lr

    def train(self, inputs, rewards):
        # Assume module has attribute prob_upper (Tunable)
        total_reward_upper = 0
        count_upper = 0
        total_reward_lower = 0
        count_lower = 0
        for ((text, action), reward) in zip(inputs, rewards):
            if action == 'upper':
                total_reward_upper += reward
                count_upper += 1
            else:
                total_reward_lower += reward
                count_lower += 1
        avg_upper = total_reward_upper / count_upper if count_upper else 0
        avg_lower = total_reward_lower / count_lower if count_lower else 0
        prob = 0.5
        if avg_upper != avg_lower:
            prob = (avg_upper - avg_lower) / (2 * abs(avg_upper - avg_lower)) + 0.5
        self.module.prob_upper.set(max(0, min(1, prob)))
