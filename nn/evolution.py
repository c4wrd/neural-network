from nn.mlff import MLFFNetwork


class EvolutionStrategy:

    def run_generation(self):
        raise NotImplementedError()

    def get_fittest_individual(self) -> MLFFNetwork:
        raise NotImplementedError()