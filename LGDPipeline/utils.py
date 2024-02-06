class IndexTensorPair:
    def __init__(self, index, tensor):
        self.index = index
        self.tensor = tensor.to('cuda')