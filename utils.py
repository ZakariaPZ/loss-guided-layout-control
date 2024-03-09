class IndexTensorPair:
    def __init__(self, idx, tensor):
        self.idx = idx
        self.tensor = tensor.to('cuda')