from enum import Enum

class ClassEnum(Enum):
    COMPLIANCE = (1, 2)
    RISK = (2, 2)
    COMPLIANCE_AND_RISK = (3, 4)

    def __init__(self, index, result_classes):
        self.index = index
        self.result_classes = result_classes