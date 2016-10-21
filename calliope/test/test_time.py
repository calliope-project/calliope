import pytest  # pylint: disable=unused-import

# import pandas as pd

# import calliope


# class TestMaskWhereZero:
#     @pytest.fixture(scope='module')
#     def testdata(self):
#         d = calliope.utils.AttrDict()
#         df1 = pd.DataFrame({'a': list(range(1, 11)),
#                             'b': list(range(20, 30))})
#         df2 = pd.DataFrame({'a': [0, 0, 0, 0, 1, 2, 3, 4, 0, 0],
#                             'b': [0, 0, 0, 1, 2, 1, 1, 2, 0, 0]})
#         df3 = pd.DataFrame({'a': [0, 0, 0, 0, 1, 2, 3, 4, 1, 0],
#                             'b': [0, 0, 0, 1, 2, 1, 1, 2, 0, 0]})
#         df4 = pd.DataFrame({'a': [0, 0, 0, 0, 1, 2, 3, 4, 1, 2],
#                             'b': [0, 0, 0, 1, 2, 1, 1, 2, 0, 0]})
#         df5 = pd.DataFrame({'a': [1, 2, 2, 3, 1, 2, 0, 0, 0, 0],
#                             'b': [0, 0, 0, 1, 2, 1, 1, 0, 0, 0]})
#         d.set_key('r.tech1', df1)
#         d.set_key('r.tech2', df2)
#         d.set_key('r.tech3', df3)
#         d.set_key('r.tech4', df4)
#         d.set_key('r.tech5', df5)
#         d.set_key('e_eff.tech1', df1 / 100.0)
#         return d

#     def test_mask_zero_no_zeros(self, testdata):
#         data = testdata
#         mask = calliope.time_masks.mask_zero(data, 'tech1')
#         assert mask.tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

#     def test_mask_zero_start_and_end_zero(self, testdata):
#         data = testdata
#         mask = calliope.time_masks.mask_zero(data, 'tech2')
#         assert mask.tolist() == [1, 1, 1, 0, 0, 0, 0, 0, 1, 1]
#         res = calliope.time_tools.masks_to_resolution_series([mask])
#         assert res.tolist() == [3, -1, -1, 0, 0, 0, 0, 0, 2, -1]

#     def test_mask_zero_one_zero(self, testdata):
#         data = testdata
#         mask = calliope.time_masks.mask_zero(data, 'tech3')
#         assert mask.tolist() == [1, 1, 1, 0, 0, 0, 0, 0, 0, 1]
#         res = calliope.time_tools.masks_to_resolution_series([mask])
#         assert res.tolist() == [3, -1, -1, 0, 0, 0, 0, 0, 0, 0]

#     def test_mask_zero_start_zero(self, testdata):
#         data = testdata
#         mask = calliope.time_masks.mask_zero(data, 'tech4')
#         assert mask.tolist() == [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
#         res = calliope.time_tools.masks_to_resolution_series([mask])
#         assert res.tolist() == [3, -1, -1, 0, 0, 0, 0, 0, 0, 0]

#     def test_mask_zero_end_zero(self, testdata):
#         data = testdata
#         mask = calliope.time_masks.mask_zero(data, 'tech5')
#         assert mask.tolist() == [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
#         res = calliope.time_tools.masks_to_resolution_series([mask])
#         assert res.tolist() == [0, 0, 0, 0, 0, 0, 0, 3, -1, -1]


# class TestDownsampling:
#     def test_infinity_test_no_inf(self):
#         pass

#     def test_infinity_test_all_inf(self):
#         pass

#     def test_infinity_test_mixed_inf(self):
#         pass

#     def test_reduce_average(self, df):
#         pass

#     def test_reduce_sum(self, df):
#         pass

#     def test_reduce_cut(self, df):
#         pass
