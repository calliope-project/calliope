import os
import pandas as pd
import pdb


from calliope.test.common.util import (
    build_test_model,
)

_MODEL_NATIONAL = os.path.join(
    os.path.dirname(__file__), "..", "example_models",
    "national_scale", "model.yaml"
)

_MODEL_URBAN = os.path.join(
    os.path.dirname(__file__), "..", "example_models",
    "urban_scale", "model.yaml"
)


def _dev_test():
    window_horizon_list = [
        (24, 24), (24, 48), (24, 72), (24, 96), (24, 120), (24, 144),
        (48, 48), (48, 72), (48, 96), (48, 120), (48, 144),
        (72, 72), (72, 96), (72, 120), (72, 144),
        (96, 96), (96, 120), (96, 144),
        (120, 120), (120, 144),
        (144, 144), (144, 168)
    ]
    # window_horizon_list = [(24, 72)]

    results = pd.DataFrame(columns=['window', 'horizon', 'result'])
    for i, (window, horizon) in enumerate(window_horizon_list):
        override_dict = {'run.operation.horizon': horizon,
                         'run.operation.window': window}
        try:
            model = build_test_model(model_file="model_operate.yaml",
                                     override_dict=override_dict)
            model.run()
            results.loc[i] = [window, horizon, 'OK']
        except IndexError:
            results.loc[i] = [window, horizon, 'IndexError']
        except ValueError:
            results.loc[i] = [window, horizon, 'ValueError']

    print(results)


if __name__ == '__main__':
    _dev_test()
