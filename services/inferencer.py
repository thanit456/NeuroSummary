from abc import ABC, abstractmethod
 
class Inferencer(ABC):

    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def infer(self, content):
        raise NotImplementedError

    @abstractmethod
    def get_name(self):
        return 'Untitled'