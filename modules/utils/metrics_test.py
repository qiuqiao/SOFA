from typing import List, Tuple

import textgrid as tg

from modules.utils.metrics import VlabelerEditDistance, VlabelerEditRatio


def point_tier_from_list(list: List[Tuple[float, str]], name="") -> tg.PointTier:
    tier = tg.PointTier(name)
    for time, mark in list:
        tier.add(time, mark)
    return tier


def get_vlabeler_edit_ratio(pred_tier, target_tier):
    dist = VlabelerEditDistance(20)
    dist.update(pred_tier, target_tier)

    ratio = VlabelerEditRatio(20)
    ratio.update(pred_tier, target_tier)
    return ratio.compute(), dist.compute()


class Test_get_vlabeler_edit_ratio:
    # 测试用例 1：完全相同
    def test_same(self):
        pred_tier = point_tier_from_list([(0, "a"), (100, "b"), (200, "")])
        target_tier = point_tier_from_list([(0, "a"), (100, "b"), (200, "")])
        edit_ratio, edit_num = get_vlabeler_edit_ratio(pred_tier, target_tier)
        assert edit_ratio == 0.0
        assert edit_num == 0

    # 测试用例 2：插入边界
    def test_insert(self):
        pred_tier = point_tier_from_list([(0, "a"), (100, "")])
        target_tier = point_tier_from_list([(0, "a"), (50, "a"), (100, "")])
        edit_ratio, edit_num = get_vlabeler_edit_ratio(pred_tier, target_tier)
        assert edit_ratio == 1 / 3
        assert edit_num == 1

        pred_tier = point_tier_from_list([(0, "a"), (100, "")])
        target_tier = point_tier_from_list([(0, "a"), (50, "b"), (100, "")])
        edit_ratio, edit_num = get_vlabeler_edit_ratio(pred_tier, target_tier)
        assert edit_ratio == 2 / 3
        assert edit_num == 2

    # 测试用例 3：删除边界
    def test_delete(self):
        pred_tier = point_tier_from_list([(0, "a"), (50, "b"), (100, "")])
        target_tier = point_tier_from_list([(0, "a"), (100, "")])
        edit_ratio, edit_num = get_vlabeler_edit_ratio(pred_tier, target_tier)
        assert edit_ratio == 1.0
        assert edit_num == 1, f"{edit_num}!= 1"

    # 测试用例 4：移动边界
    def test_move(self):
        pred_tier = point_tier_from_list([(0, "a"), (100, "b"), (200, "")])
        target_tier = point_tier_from_list([(0, "a"), (121, "b"), (200, "")])
        edit_ratio, edit_num = get_vlabeler_edit_ratio(pred_tier, target_tier)
        assert edit_ratio == 1 / 3
        assert edit_num == 1

        pred_tier = point_tier_from_list([(0, "a"), (100, "b"), (200, "")])
        target_tier = point_tier_from_list([(0, "a"), (120, "b"), (200, "")])
        edit_ratio, edit_num = get_vlabeler_edit_ratio(pred_tier, target_tier)
        assert edit_ratio == 0.0
        assert edit_num == 0

    # 测试用例 5：音素替换
    def test_replace(self):
        pred_tier = point_tier_from_list([(0, "a"), (100, "b"), (200, "")])
        target_tier = point_tier_from_list([(0, "b"), (100, "c"), (200, "")])
        edit_ratio, edit_num = get_vlabeler_edit_ratio(pred_tier, target_tier)
        assert edit_ratio == 2 / 3
        assert edit_num == 2
