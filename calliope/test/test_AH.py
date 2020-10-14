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
    horizon_window_list = [
        (24, 24), (48, 24), (72, 24), (96, 24), (48, 48), (72, 48), (96, 48),
        (72, 72), (96, 72), (96, 96),
    ]

    results = pd.DataFrame(columns=['horizon', 'window', 'result'])
    for i, (horizon, window) in enumerate(horizon_window_list):
        override_dict = {'run.operation.horizon': horizon,
                         'run.operation.window': window}
        try:
            model = build_test_model(model_file="model_operate.yaml",
                                     override_dict=override_dict)
            model.run()
            results.loc[i] = [horizon, window, 'OK']
        except IndexError:
            results.loc[i] = [horizon, window, 'IndexError']
        except ValueError:
            results.loc[i] = [horizon, window, 'ValueError']

    print(results)


if __name__ == '__main__':
    _dev_test()
