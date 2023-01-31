

class GA:
    def __init__(self, max_stages): # max_stage_len
        self.max_stages = max_stages# amount of unique regularizers
        raise NotImplementedError

    def init_individ(self):
        raise NotImplementedError
