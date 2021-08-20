class Result:
    def __init__(self, method_name, data_name, alg, alg_kwargs, data_kwargs,
                 p_values, alternates, rejsets, instances):
        self.method_name = method_name
        self.data_name = data_name
        self.alg = alg
        self.alg_kwargs = alg_kwargs
        self.data_kwargs = data_kwargs
        self.p_values = p_values
        self.alternates = alternates
        self.trials, self.hypotheses = p_values.shape
        self.rejsets = rejsets
        self.instances = instances
