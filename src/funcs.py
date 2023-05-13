from collections.abc import MutableSequence

class stack(list):
    '''Simplifies life'''
    def __init__(self):
        super().__init__()

    def pop_second_to_top(self):
        self.pop(-2)

    def pop_top(self):
        self.pop()

class buffer(list):
    def __init__(self):
        super().__init__()
    
    def get_next(self):
        x = self.pop(0)
        return x