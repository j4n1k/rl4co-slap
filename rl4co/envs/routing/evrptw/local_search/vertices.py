

class Vertice:
    def __init__(self, vertex_id: int):
        self.vertex_id = id


class Depot(Vertice):
    """

    """
    def __init__(self, vertex_id: int):
        super().__init__(vertex_id)


class Customer(Vertice):
    def __init__(self, vertex_id: int):
        super().__init__(vertex_id)
