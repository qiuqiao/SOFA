from typing import List, Tuple

import textgrid as tg

from modules.utils.metrics import (
    BoundaryEditDistance,
    BoundaryEditRatio,
    IntersectionOverUnion,
    VlabelerEditRatio,
    VlabelerEditsCount,
)


def point_tier_from_list(list: List[Tuple[float, str]], name="") -> tg.PointTier:
    tier = tg.PointTier(name)
    for time, mark in list:
        tier.add(time, mark)
    return tier


def get_vlabeler_edit_ratio(pred_tier, target_tier):
    dist = VlabelerEditsCount(move_tolerance=20)
    dist.update(pred_tier, target_tier)

    ratio = VlabelerEditRatio(move_tolerance=20)
    ratio.update(pred_tier, target_tier)
    return ratio.compute(), dist.compute()


class TestVlabelerEditRatio:
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
        assert edit_ratio == round(1 / 3, 6)
        assert edit_num == 1

        pred_tier = point_tier_from_list([(0, "a"), (100, "")])
        target_tier = point_tier_from_list([(0, "a"), (50, "b"), (100, "")])
        edit_ratio, edit_num = get_vlabeler_edit_ratio(pred_tier, target_tier)
        assert edit_ratio == round(2 / 3, 6)
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
        assert edit_ratio == round(1 / 3, 6)
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
        assert edit_ratio == round(2 / 3, 6)
        assert edit_num == 2


class TestIntersectionOverUnion:
    def test_same(self):
        pred_tier = point_tier_from_list([(0, "a"), (100, "b"), (200, "")])
        target_tier = point_tier_from_list([(0, "a"), (100, "b"), (200, "")])
        answer = {
            "a": 1.0,
            "b": 1.0,
        }

        iou = IntersectionOverUnion()
        iou.update(pred_tier, target_tier)
        pred_answer = iou.compute()

        assert pred_answer == answer, f"{pred_answer}!= {answer}"

    def test_zero(self):
        pred_tier = point_tier_from_list([(0, "a"), (50, "b"), (250, "")])
        target_tier = point_tier_from_list([(0, "c"), (150, "d"), (200, "")])
        answer = {
            "a": 0.0,
            "b": 0.0,
            "c": 0.0,
            "d": 0.0,
        }

        iou = IntersectionOverUnion()
        iou.update(pred_tier, target_tier)
        pred_answer = iou.compute()

        assert pred_answer == answer, f"{pred_answer}!= {answer}"

        pred_tier = point_tier_from_list([(0, "a"), (100, "b"), (200, "")])
        target_tier = point_tier_from_list([(0, "b"), (100, "a"), (200, "")])
        answer = {
            "a": 0.0,
            "b": 0.0,
        }

        iou.reset()
        iou.update(pred_tier, target_tier)
        pred_answer = iou.compute()

        assert pred_answer == answer, f"{pred_answer}!= {answer}"

    def test_half(self):
        pred_tier = point_tier_from_list([(0, "a"), (50, "b"), (250, "")])
        target_tier = point_tier_from_list([(0, "a"), (100, "b"), (200, "")])
        answer = {
            "a": 0.5,
            "b": 0.5,
        }

        iou = IntersectionOverUnion()
        iou.update(pred_tier, target_tier)
        pred_answer = iou.compute()

        assert pred_answer == answer, f"{pred_answer}!= {answer}"


class TestBoundaryEditDistance:
    def test_same(self):
        pred_tier = point_tier_from_list([(0, "a"), (100, "b"), (200, "")])
        target_tier = point_tier_from_list([(0, "a"), (100, "b"), (200, "")])
        answer = 0.0

        metric = BoundaryEditDistance()
        metric.update(pred_tier, target_tier)
        pred_answer = metric.compute()

        assert pred_answer == answer, f"{pred_answer}!= {answer}"

    def test_2(self):
        pred_tier = point_tier_from_list([(0, "a"), (100, "b"), (200, "")])
        target_tier = point_tier_from_list([(0, "a"), (150, "b"), (200, "")])
        answer = 50.0

        metric = BoundaryEditDistance()
        metric.update(pred_tier, target_tier)
        pred_answer = metric.compute()

        assert pred_answer == answer, f"{pred_answer}!= {answer}"

    def test_3(self):
        pred_tier = point_tier_from_list([(0, "a"), (100, "b"), (200, "")])
        target_tier = point_tier_from_list([(50, "a"), (100, "b"), (150, "")])
        answer = 100.0

        metric = BoundaryEditDistance()
        metric.update(pred_tier, target_tier)
        pred_answer = metric.compute()

        assert pred_answer == answer, f"{pred_answer}!= {answer}"

    def test_assert(self):
        pred_tier = point_tier_from_list([(0, "a"), (100, "b"), (200, "")])
        target_tier = point_tier_from_list([(0, "a"), (100, "b"), (200, "c")])

        metric = BoundaryEditDistance()
        try:
            metric.update(pred_tier, target_tier)
        except AssertionError:
            return
        assert False, "AssertionError not raised"


class TestBoundaryEditRatio:
    def test_same(self):
        pred_tier = point_tier_from_list([(0, "a"), (100, "b"), (200, "")])
        target_tier = point_tier_from_list([(0, "a"), (100, "b"), (200, "")])
        answer = 0.0

        metric = BoundaryEditRatio()
        metric.update(pred_tier, target_tier)
        pred_answer = metric.compute()

        assert pred_answer == answer, f"{pred_answer}!= {answer}"

    def test_2(self):
        pred_tier = point_tier_from_list([(0, "a"), (100, "b"), (200, "")])
        target_tier = point_tier_from_list([(0, "a"), (150, "b"), (200, "")])
        answer = 50.0 / 200.0

        metric = BoundaryEditRatio()
        metric.update(pred_tier, target_tier)
        pred_answer = metric.compute()

        assert pred_answer == answer, f"{pred_answer}!= {answer}"

    def test_3(self):
        pred_tier = point_tier_from_list([(0, "a"), (100, "b"), (200, "")])
        target_tier = point_tier_from_list([(50, "a"), (100, "b"), (150, "")])
        answer = 100.0 / (150.0 - 50.0)

        metric = BoundaryEditRatio()
        metric.update(pred_tier, target_tier)
        pred_answer = metric.compute()

        assert pred_answer == answer, f"{pred_answer}!= {answer}"

    def test_assert(self):
        pred_tier = point_tier_from_list([(0, "a"), (100, "b"), (200, "")])
        target_tier = point_tier_from_list([(0, "a"), (100, "b"), (200, "c")])

        metric = BoundaryEditRatio()
        try:
            metric.update(pred_tier, target_tier)
        except AssertionError:
            return
        assert False, "AssertionError not raised"
