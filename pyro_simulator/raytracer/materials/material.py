from abc import abstractmethod

class Material:
    def __init__(self):
        pass

    def get_Normal(self, hit):
        N_coll = hit.collider.get_Normal(hit)
        return N_coll * hit.orientation.reshape(-1, 1)

    @abstractmethod
    def get_color(self, scene, ray, hit, max_index):
        pass
