def inheritance(model_data, **kwargs):
    def _inheritance(tech_group):
        # Only for base tech inheritance
        return model_data.inheritance.str.endswith(tech_group)

    return _inheritance


def backend_sum(backend_interface, **kwargs):
    def _backend_sum(to_sum, *, over):
        return backend_interface.sum(to_sum)

    return _backend_sum
