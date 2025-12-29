#测试用的
import unittest
# 导入被测函数，就像 Java 的 import
from tsp import minout, inf, backtracking, distance , two_minout

class TSPMinOutTest(unittest.TestCase):
    
    # 对应 Java 的 @BeforeEach
    def setUp(self):
        """Initialize the graph from the image."""
        self.graph = [
            [inf, 1,   3,   5,   8],   # A (0)
            [1,   inf, 4,   2,   9],   # B (1)
            [3,   4,   inf, 7,   2],   # C (2)
            [5,   2,   7,   inf, 2],   # D (3)
            [8,   9,   2,   2,   inf]  # E (4)
        ]

    # 对应 Java 的 @Test
    def test_root_node_bound(self):
        """Test Root Node A purple number: Expected 8."""
        self.assertEqual(minout([0], self.graph), 8)

    def test_path_ab_bound(self):
        """Test Path A->B purple number: Expected 8."""
        self.assertEqual(minout([0, 1], self.graph), 8)

    def test_path_abd_bound(self):
        """Test Path A->B->D: Should be 9 (Correcting the image error)."""
        # 已走 1+2=3, 剩余 D,C,E 最小出边 2+2+2=6, 总计 9
        self.assertEqual(minout([0, 1, 3], self.graph), 9)

    def test_path_ac_bound(self):
        """Test Path A->C purple number: Expected 10."""
        self.assertEqual(minout([0, 2], self.graph), 10)
    

    def test_solve_full_tsp(self):
        """
        测试整体回溯搜索是否能找到图片中的最优路径
        期望结果：ABDEC (0-1-3-4-2-0), 距离 10
        """
        # 使用 minout 作为下界函数进行分支界限搜索
        best_cost, best_tour = backtracking(self.graph, [0], inf, [])
        
        # 验证路径非空
        self.assertIsNotNone(best_tour)
        # 验证总距离是否为 10
        self.assertEqual(distance(best_tour, self.graph), 10)
        # 验证路径序列是否正确
        self.assertEqual(best_tour, [0, 1, 3, 4, 2, 0])

#====================12/29更新==========================

    def test_two_minout(self):
        """
        全面测试 two_minout 的有效性，覆盖所有等价类：
        1. 边界值：根节点（第0层）
        2. 第一层分支
        3. 中间层：产生小数 (.5) 的节点（第2层、第3层）
        4. 叶子前期：第4层
        """
        # (路径, 预期 Bound)
        test_cases = [
            # --- 第0层 (Root Boundary) ---
            ([0], 10.0),

            # --- 第1层 (First Level Branches) ---
            ([0, 1], 9.0),  # A->B
            ([0, 2], 11.0),  # A->C
            ([0, 3], 13.0),  # A->D
            ([0, 4], 16.0),  # A->E

            # --- 第2层 (Intermediate - Decimal Check) ---
            ([0, 1, 2], 11.5),  # A->B->C: 5 + (5+4+4)/2 = 11.5
            ([0, 1, 3], 9.5),  # A->B->D: 3 + (4+5+4)/2 = 9.5
            ([0, 1, 4], 16.5),  # A->B->E: 10 + (4+5+4)/2 = 16.5

            # --- 第3层 (Deep Search Check) ---
            ([0, 1, 3, 2], 14.5),  # A->B->D->C: 10 + (5+4)/2 = 14.5
            ([0, 1, 3, 4], 9.5),  # A->B->D->E: 5 + (4+5)/2 = 9.5

            # --- 第4层 (Near Leaf Check) ---
            ([0, 1, 3, 4, 2], 9.5)  # A->B->D->E->C: 7 + 5/2 = 9.5
        ]

        for path, expected in test_cases:
            # 使用 subTest 确保即使一个失败，其他测试也会继续，并标记出错的路径
            with self.subTest(path=path):
                actual = two_minout(path, self.graph)
                self.assertAlmostEqual(actual, expected, places=1,
                                       msg=f"Bound calculation error for path {path}!")

    def test_tsp_with_two_minout(self):
        """
        集成测试：确保 two_minout 能引导 backtracking 正确找到最优解 10
        """
        best_cost, best_tour = backtracking(self.graph, [0], inf, [], bounding=two_minout)

        self.assertIsNotNone(best_tour)
        self.assertEqual(best_cost, 10.0)
        self.assertEqual(best_tour, [0, 1, 3, 4, 2, 0])


if __name__ == '__main__':
    unittest.main()