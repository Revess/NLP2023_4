from collections.abc import MutableSequence

class stack(list):
    '''Simplifies life'''
    def __init__(self):
        super().__init__()
        self.append("ROOT")

    def pop_second_to_top(self):
        self.pop(-2)

    def pop_top(self):
        self.pop()

class buffer(list):
    def __init__(self, data=None):
        super().__init__()
        if type(data) == list:
            for entry in data:
                self.append(entry)
    
    def get_next(self):
        x = self.pop(0)
        return x