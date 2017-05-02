from enum import Enum

class ClassEnum(Enum):
    COMPLIANCE = (1, 2, 75, 0.01)
    RISK = (2, 2, 75, 0.01)
    COMPLIANCE_AND_RISK = (3, 4, 100, 0.05)

    def __init__(self, index, result_classes, hidden, rate):
        self.index = index
        self.result_classes = result_classes
        self.hidden = hidden
        self.rate = rate