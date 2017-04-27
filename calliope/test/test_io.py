import os
import tempfile

import numpy as np
import pytest

import calliope


def verify_solution_integrity(model_solution, solution_from_disk, tempdir):
        # Check whether the two are the same
        np.allclose(model_solution['e_cap'], solution_from_disk['e_cap'])

        # Check that config AttrDict has been deserialized
        assert(solution_from_disk.attrs['config_run'].output.path == tempdir)


class TestSave:
    @pytest.fixture(scope='module')
    def model(self):
        model = calliope.examples.NationalScale()
        model.run()
        return model

    def test_save_netcdf(self, model):
        with tempfile.TemporaryDirectory() as tempdir:
            model.config_run.set_key('output.path', tempdir)
            model.save_solution('netcdf')

            # Try reading solution back in
            sol_file = os.path.join(tempdir, 'solution.nc')
            solution_from_disk = calliope.read.read_netcdf(sol_file)
            solution_from_disk.close()  # so that temp dir can be deleted

        verify_solution_integrity(model.solution, solution_from_disk, tempdir)

    def test_save_csv(self, model):
        with tempfile.TemporaryDirectory() as tempdir:
            model.config_run.set_key('output.path', tempdir)
            model.save_solution('csv')

            # Try reading solution back in
            solution_from_disk = calliope.read.read_csv(tempdir)

        verify_solution_integrity(model.solution, solution_from_disk, tempdir)
