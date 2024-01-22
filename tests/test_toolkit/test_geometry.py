import math
import unittest

import numpy as np

from core.toolkit.geometry import *


class TestGeometry(unittest.TestCase):
    def test_point(self):
        origin = Point.origin()
        id_matrix = Matrix2D.identical()
        self.assertEqual(origin, Point(0, 0))
        self.assertNotEqual(origin, Point(0, 0.1))
        self.assertNotEqual(origin, (0, 0.1))
        self.assertEqual(origin + Point(1, 2), Point(1, 2))
        self.assertEqual(origin - Point(1, 2), Point(-1, -2))
        self.assertEqual(Point(1, 1).rotate(0.25 * math.pi), Point(0, math.sqrt(2)))
        self.assertEqual(id_matrix @ Point(1.2, 3.4), Point(1.2, 3.4))
        self.assertAlmostEqual(Point(1, 1).theta, 0.25 * math.pi)
        self.assertTrue(Point(0.5, 0.5).inside(id_matrix))
        self.assertFalse(Point(1.5, 1.5).inside(id_matrix))

    def test_line(self):
        l0 = Line(Point(0, 0), Point(1, 0))
        l1 = Line(Point(0, 1), Point(1, 1))
        l2 = Line(Point(0, 0), Point(1, 1))
        l3 = Line(Point(0, 0), Point(0.5, 0.5))
        self.assertIsNone(l0.intersect(l1))
        self.assertEqual(l1.intersect(l2), Point(1, 1))
        self.assertEqual(l1.intersect(l3, extendable=True), Point(1, 1))
        self.assertIsNone(l1.intersect(l3))
        self.assertAlmostEqual(l0.distance_to(l1), 1)

    def test_matrix2d(self):
        id_matrix = Matrix2D.identical()
        id_flipY_matrix = id_matrix.flip(False, True)
        self.assertNotEqual(id_matrix, 1)
        with self.assertRaises(TypeError):
            id_matrix @ 1
        self.assertEqual(id_matrix.x, 0)
        self.assertEqual(id_matrix.y, 0)
        self.assertEqual(id_matrix.position, Point.origin())
        self.assertEqual(id_matrix.translation, Point.origin())
        self.assertAlmostEqual(id_matrix.w, 1)
        self.assertAlmostEqual(id_matrix.h, 1)
        self.assertAlmostEqual(id_matrix.wh[0], 1)
        self.assertAlmostEqual(id_matrix.wh[1], 1)
        self.assertAlmostEqual(id_matrix.wh_ratio, 1)
        self.assertAlmostEqual(id_matrix.abs_wh[0], 1)
        self.assertAlmostEqual(id_matrix.abs_wh[1], 1)
        self.assertAlmostEqual(id_matrix.abs_wh_ratio, 1)
        self.assertAlmostEqual(id_flipY_matrix.wh[1], -1)
        self.assertAlmostEqual(id_flipY_matrix.abs_wh[1], 1)
        self.assertAlmostEqual(id_flipY_matrix.wh_ratio, -1)
        self.assertAlmostEqual(id_flipY_matrix.abs_wh_ratio, 1)
        self.assertAlmostEqual(id_matrix.area, 1)
        self.assertAlmostEqual(id_flipY_matrix.area, 1)
        self.assertAlmostEqual(id_matrix.theta, 0)
        self.assertAlmostEqual(id_matrix.shear, 0)
        self.assertAlmostEqual(id_matrix.determinant, 1)
        np.testing.assert_allclose(id_matrix.matrix, np.array([[1, 0, 0], [0, 1, 0]]))
        self.assertEqual(id_matrix.lt, Point(0, 0))
        self.assertEqual(id_matrix.rt, Point(1, 0))
        self.assertEqual(id_matrix.lb, Point(0, 1))
        self.assertEqual(id_matrix.rb, Point(1, 1))
        self.assertEqual(id_matrix.center, Point(0.5, 0.5))
        self.assertEqual(id_matrix.top, Point(0.5, 0))
        self.assertEqual(id_matrix.bottom, Point(0.5, 1))
        self.assertEqual(id_matrix.left, Point(0, 0.5))
        self.assertEqual(id_matrix.right, Point(1, 0.5))
        self.assertEqual(id_matrix.pivot(PivotType.CENTER), Point(0.5, 0.5))
        self.assertEqual(len(id_matrix.corner_points), 4)
        self.assertEqual(len(id_matrix.mid_points), 5)
        self.assertEqual(len(id_matrix.all_points), 9)
        self.assertEqual(len(id_matrix.edges), 4)
        self.assertEqual(id_matrix.outer_most, Box(0, 0, 1, 1))
        self.assertEqual(id_matrix.bounding, id_matrix)
        id_css = "matrix(1.0,0.0,0.0,1.0,0.0,0.0)"
        self.assertEqual(id_matrix.css_property, id_css)
        self.assertEqual(id_matrix, Matrix2D.from_css_property(id_css))
        self.assertEqual(id_matrix.no_move, id_matrix)
        self.assertEqual(id_matrix.no_skew, id_matrix)
        self.assertEqual(id_matrix.no_scale, id_matrix)
        self.assertEqual(id_matrix.no_scale_but_flip, id_matrix)
        self.assertEqual(id_matrix.no_rotation, id_matrix)
        self.assertEqual(id_matrix.no_move_scale_but_flip, id_matrix)
        double_matrix = Matrix2D(a=2, b=0, c=0, d=2, e=0, f=0)
        self.assertEqual(id_matrix.scale(2), double_matrix)
        self.assertEqual(id_matrix.scale(2, center=Point.origin()), double_matrix)
        self.assertEqual(id_matrix.scale_to(2), double_matrix)
        self.assertEqual(id_matrix.rotate(2 * math.pi), id_matrix)
        self.assertEqual(id_matrix.rotate_to(0), id_matrix)
        self.assertEqual(id_matrix.set_w(2).set_h(2), double_matrix)
        self.assertEqual(id_matrix.set_wh(2, 2), double_matrix)
        unit_trans_matrix = Matrix2D(a=1, b=0, c=0, d=1, e=1, f=1)
        self.assertEqual(id_matrix.move(Point(1, 1)), unit_trans_matrix)
        self.assertEqual(id_matrix.move_to(Point(1, 1)), unit_trans_matrix)
        self.assertEqual(Matrix2D.get_bounding_of([]), id_matrix)
        self.assertEqual(
            Matrix2D.get_bounding_of([id_matrix, double_matrix]), double_matrix
        )
        self.assertEqual(
            id_matrix.set_wh_ratio(2, type=ExpandType.FIX_W, pivot=PivotType.LT),
            Matrix2D(a=1, b=0, c=0, d=0.5, e=0, f=0),
        )
        self.assertEqual(
            id_matrix.set_wh_ratio(2, type=ExpandType.FIX_H, pivot=PivotType.LT),
            Matrix2D(a=2, b=0, c=0, d=1, e=0, f=0),
        )
        self.assertEqual(
            id_matrix.set_wh_ratio(2, type=ExpandType.IOU, pivot=PivotType.LT),
            Matrix2D(a=math.sqrt(2), b=0, c=0, d=0.5 * math.sqrt(2), e=0, f=0),
        )

    def test_hit_test(self):
        l0 = Line(Point(0, 0), Point(1, 0))
        l1 = Line(Point(0, 1), Point(1, 1))
        l2 = Line(Point(0, 0), Point(1, 1))
        l3 = Line(Point(0.25, 0.25), Point(0.75, 0.75))
        self.assertFalse(HitTest.line_line(l0, l1))
        self.assertTrue(HitTest.line_line(l1, l2))
        self.assertFalse(HitTest.line_line(l1, l3))
        id_matrix = Matrix2D.identical()
        self.assertTrue(HitTest.line_box(l0, id_matrix))
        self.assertTrue(HitTest.line_box(l1, id_matrix))
        self.assertTrue(HitTest.line_box(l2, id_matrix))
        self.assertFalse(HitTest.line_box(l3, id_matrix))
        self.assertTrue(HitTest.box_box(id_matrix, id_matrix))
        self.assertTrue(
            HitTest.box_box(id_matrix, Matrix2D(a=1, b=0, c=0, d=1, e=0.5, f=0.5))
        )
        self.assertTrue(
            HitTest.box_box(id_matrix, Matrix2D(a=3, b=0, c=0, d=3, e=-1, f=-1))
        )
        self.assertTrue(
            HitTest.box_box(Matrix2D(a=3, b=0, c=0, d=3, e=-1, f=-1), id_matrix)
        )
        self.assertFalse(
            HitTest.box_box(id_matrix, Matrix2D(a=1, b=0, c=0, d=1, e=1.5, f=1.5))
        )


if __name__ == "__main__":
    unittest.main()
