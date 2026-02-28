import numpy as np
import open3d as o3d
import os
import glob
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json
from functools import lru_cache


@dataclass
class MatchResult:
    """匹配结果数据结构"""
    cad_file: str
    fitness: float  # 重叠度 (0-1)
    inlier_rmse: float  # 内点均方根误差
    transformation: np.ndarray  # 4x4变换矩阵
    correspondence_distance: float  # 对应点距离
    processing_time: float
    
    def __repr__(self):
        return (f"MatchResult(cad={Path(self.cad_file).name}, "
                f"fitness={self.fitness:.4f}, RMSE={self.inlier_rmse:.4f}mm)")


class HighPrecisionPointCloudMatcher:
    """
    高精度点云匹配系统
    适用于牙科口扫数据与CAD模型的匹配
    """
    
    def __init__(self, 
                 voxel_size: float = 0.1,  # 下采样体素大小(mm)
                 normal_radius: float = 0.3,  # 法向量估计半径
                 feature_radius: float = 0.5,  # FPFH特征半径
                 icp_threshold: float = 0.5,  # ICP对应点最大距离
                 n_samples: int = 3,  # 粗匹配候选数量
                 use_teaser: bool = False,  # 默认关闭TEASER避免长时间卡住
                 use_fgr: bool = False,  # 默认关闭FGR避免额外耗时
                 sample_points: int = 5000,  # 更小采样加速
                 cache_cad: bool = False,  # 是否缓存CAD特征，默认关闭以节省内存
                 max_preload: int = 50):  # 预加载CAD上限，避免一次性占满内存
        """
        参数针对牙科口扫优化（单位：mm）
        """
        self.voxel_size = voxel_size
        # 直接使用外部传入的物理尺寸（不再与voxel_size相乘），避免半径过小导致匹配对应点过少
        self.normal_radius = normal_radius
        self.feature_radius = feature_radius
        self.icp_threshold = icp_threshold
        self.n_samples = n_samples
        self.use_teaser = use_teaser
        self.use_fgr = use_fgr
        self.sample_points = sample_points
        self.cache_cad = cache_cad
        self.max_preload = max_preload
        
        # 缓存CAD特征避免重复计算
        self._cad_cache = {}
        self._teaser_available = self._check_teaser()

    @staticmethod
    @lru_cache(maxsize=1)
    def _check_teaser() -> bool:
        try:
            import teaserpp_python  # noqa: F401
            return True
        except Exception:
            return False
        
    def _load_stl(self, file_path: str) -> o3d.geometry.PointCloud:
        """加载STL文件并转换为点云"""
        mesh = o3d.io.read_triangle_mesh(file_path)
        if mesh.is_empty():
            raise ValueError(f"无法加载文件: {file_path}")
        
        # 确保法向量正确
        mesh.compute_vertex_normals()
        
        # 从网格采样点云（均匀采样保证精度）
        pcd = mesh.sample_points_uniformly(number_of_points=self.sample_points)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.normal_radius * 2, max_nn=30
            )
        )
        
        # 中心化
        pcd_center = pcd.get_center()
        pcd.translate(-pcd_center)
        
        return pcd
    
    def _preprocess(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """预处理：下采样、去噪、法向量估计"""
        # 统计滤波去噪
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0
        )
        
        # 体素下采样
        pcd_down = pcd.voxel_down_sample(self.voxel_size)
        
        # 估计法向量（对配准至关重要）
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.normal_radius, max_nn=30
            )
        )
        
        # 法向量定向（朝向视点）
        pcd_down.orient_normals_consistent_tangent_plane(100)
        
        return pcd_down
    
    def _compute_fpfh(self, pcd: o3d.geometry.PointCloud) -> o3d.pipelines.registration.Feature:
        """计算FPFH特征描述子"""
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.feature_radius, max_nn=100
            )
        )
        return fpfh
    
    def _global_registration(self, 
                           source: o3d.geometry.PointCloud,
                           target: o3d.geometry.PointCloud,
                           source_fpfh: o3d.pipelines.registration.Feature,
                           target_fpfh: o3d.pipelines.registration.Feature) -> o3d.pipelines.registration.RegistrationResult:
        """
        基于FPFH的全局配准。优先 TEASER++，其次 FGR，最后 RANSAC。
        """
        if self.use_teaser and self._teaser_available:
            teaser_result = self._global_registration_teaser(source, target, source_fpfh, target_fpfh)
            if teaser_result is not None:
                return teaser_result

        if self.use_fgr:
            fgr_result = self._global_registration_fgr(source, target, source_fpfh, target_fpfh)
            if fgr_result is not None:
                return fgr_result

        distance_threshold = self.feature_radius * 1.5
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source, target, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )
        return result

    def _global_registration_fgr(self,
                                 source: o3d.geometry.PointCloud,
                                 target: o3d.geometry.PointCloud,
                                 source_fpfh: o3d.pipelines.registration.Feature,
                                 target_fpfh: o3d.pipelines.registration.Feature) -> Optional[o3d.pipelines.registration.RegistrationResult]:
        """快速全局配准(FGR)作为 TEASER++ 失败时的回退。"""
        try:
            option = o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=self.feature_radius * 1.5,
                iteration_number=32
            )
            result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                source, target, source_fpfh, target_fpfh, option)
            if result.transformation.trace() == 4.0:
                return None
            return result
        except Exception:
            return None

    def _global_registration_teaser(self,
                                    source: o3d.geometry.PointCloud,
                                    target: o3d.geometry.PointCloud,
                                    source_fpfh: o3d.pipelines.registration.Feature,
                                    target_fpfh: o3d.pipelines.registration.Feature) -> Optional[o3d.pipelines.registration.RegistrationResult]:
        """使用 TEASER++ 做鲁棒全局配准，增加特征对应数量并过滤。"""
        try:
            import teaserpp_python as teaser
        except Exception:
            return None

        src_feat = np.asarray(source_fpfh.data).T  # [N, 33]
        tgt_feat = np.asarray(target_fpfh.data).T
        if src_feat.size == 0 or tgt_feat.size == 0:
            return None

        # kNN 双向匹配，较小 k 加速
        from scipy.spatial import cKDTree
        k = 2
        src_tree = cKDTree(src_feat)
        tgt_tree = cKDTree(tgt_feat)
        _, tgt_idx = tgt_tree.query(src_feat, k=k)
        _, src_idx = src_tree.query(tgt_feat, k=k)

        correspondences = []
        for s, tgt_neighbors in enumerate(tgt_idx):
            for t in np.atleast_1d(tgt_neighbors):
                if s in src_idx[int(t)]:  # mutual in kNN
                    correspondences.append((s, int(t)))

        # 去重
        correspondences = list({(s, t) for s, t in correspondences})
        if len(correspondences) < 8:
            return None

        src_points = np.asarray(source.points)
        tgt_points = np.asarray(target.points)
        src_corr = src_points[[c[0] for c in correspondences]].T  # 3 x N
        tgt_corr = tgt_points[[c[1] for c in correspondences]].T

        params = teaser.RobustRegistrationSolver.Params()
        params.cbar2 = 1
        params.noise_bound = max(self.icp_threshold, 1e-3)
        params.estimate_scaling = False
        params.rotation_max_iterations = 50
        params.rotation_gnc_factor = 1.4
        params.rotation_estimation_algorithm = teaser.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS

        solver = teaser.RobustRegistrationSolver(params)
        solver.solve(src_corr, tgt_corr)
        sol = solver.getSolution()

        transform = np.eye(4)
        transform[:3, :3] = sol.rotation
        transform[:3, 3] = sol.translation

        eval_result = o3d.pipelines.registration.evaluate_registration(
            source, target, max_correspondence_distance=self.icp_threshold, transformation=transform)
        eval_result.transformation = transform
        return eval_result
    
    def _local_refinement(self,
                         source: o3d.geometry.PointCloud,
                         target: o3d.geometry.PointCloud,
                         init_transform: np.ndarray) -> o3d.pipelines.registration.RegistrationResult:
        """
        基于点到平面的ICP精配准
        针对高精度牙科数据优化
        """
        # 多尺度ICP策略
        thresholds = [self.icp_threshold * 4, self.icp_threshold * 2, self.icp_threshold]
        current_transform = init_transform
        
        for threshold in thresholds:
            result = o3d.pipelines.registration.registration_icp(
                source, target, threshold, current_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=200,
                    relative_fitness=1e-6,
                    relative_rmse=1e-6
                )
            )
            current_transform = result.transformation
            
        return result
    
    def _evaluate_symmetry(self, 
                          source: o3d.geometry.PointCloud,
                          target: o3d.geometry.PointCloud,
                          transform: np.ndarray) -> float:
        """
        评估双侧对称性（针对牙科应用）
        返回对称性评分 (0-1, 越高越好)
        """
        source_transformed = source.transform(transform)
        points = np.asarray(source_transformed.points)
        
        # 假设X轴为对称轴，计算对称性
        if len(points) == 0:
            return 0.0
            
        mirrored = points.copy()
        mirrored[:, 0] = -mirrored[:, 0]  # X轴镜像
        
        # 构建KDTree查找对应点
        target_tree = o3d.geometry.KDTreeFlann(target)
        symmetry_errors = []
        
        for pt in mirrored:
            [_, idx, dist] = target_tree.search_knn_vector_3d(pt, 1)
            symmetry_errors.append(np.sqrt(dist[0]))
            
        # 转换为评分
        mean_error = np.mean(symmetry_errors)
        score = max(0, 1.0 - mean_error / (self.voxel_size * 10))
        
        return score
    
    def match_single(self, 
                    scan_file: str, 
                    cad_file: str,
                    visualize: bool = False) -> MatchResult:
        """
        单对匹配：扫描数据 vs CAD模型
        """
        import time
        start_time = time.time()
        
        # 1. 加载数据
        # print(f"加载扫描数据: {scan_file}")
        scan_pcd = self._load_stl(scan_file)
        # print(f"  点数: {len(scan_pcd.points)}")
        
        print(f"加载CAD模型: {cad_file}")
        if cad_file in self._cad_cache:
            cad_pcd, cad_fpfh = self._cad_cache[cad_file]
        else:
            cad_pcd = self._load_stl(cad_file)
            cad_pcd_prep = self._preprocess(cad_pcd)
            cad_fpfh = self._compute_fpfh(cad_pcd_prep)
            if self.cache_cad:
                self._cad_cache[cad_file] = (cad_pcd_prep, cad_fpfh)
            cad_pcd = cad_pcd_prep
        # print(f"  点数: {len(cad_pcd.points)}")
        
        # 2. 预处理扫描数据
        scan_prep = self._preprocess(scan_pcd)
        scan_fpfh = self._compute_fpfh(scan_prep)
        
        # 3. 全局配准（粗匹配）
        # print("执行全局配准 (TEASER++ 优先)...")
        global_result = self._global_registration(scan_prep, cad_pcd, scan_fpfh, cad_fpfh)
        print(f"  粗匹配 fitness: {global_result.fitness:.4f}")
        
        # 4. 局部精配准
        # print("执行ICP精配准...")
        icp_result = self._local_refinement(scan_prep, cad_pcd, global_result.transformation)
        print(f"  精匹配 fitness: {icp_result.fitness:.4f}, RMSE: {icp_result.inlier_rmse:.4f}")
        
        # 5. 计算对称性评分（牙科特异性）
        symmetry_score = self._evaluate_symmetry(scan_prep, cad_pcd, icp_result.transformation)
        # print(f"  对称性评分: {symmetry_score:.4f}")
        
        # 6. 综合评分
        combined_score = (icp_result.fitness * 0.6 + 
                         (1 - min(icp_result.inlier_rmse / self.voxel_size, 1)) * 0.3 +
                         symmetry_score * 0.1)
        
        processing_time = time.time() - start_time
        
        result = MatchResult(
            cad_file=cad_file,
            fitness=icp_result.fitness,
            inlier_rmse=icp_result.inlier_rmse,
            transformation=icp_result.transformation,
            correspondence_distance=combined_score,
            processing_time=processing_time
        )
        
        # 可视化
        if visualize:
            self._visualize_result(scan_prep, cad_pcd, icp_result.transformation, result)
            
        return result
    
    def match_batch(self, 
                   scan_folder: str, 
                   cad_folder: str,
                   output_json: Optional[str] = None) -> List[MatchResult]:
        """
        批量匹配：从scan文件夹选取扫描数据，匹配cad文件夹中所有模型
        返回最佳匹配
        """
        scan_files = glob.glob(os.path.join(scan_folder, "*.stl"))
        cad_files = glob.glob(os.path.join(cad_folder, "*.stl"))
        
        if not scan_files:
            raise FileNotFoundError(f"在 {scan_folder} 中未找到STL文件")
        if not cad_files:
            raise FileNotFoundError(f"在 {cad_folder} 中未找到STL文件")
        
        print(f"发现 {len(scan_files)} 个扫描文件, {len(cad_files)} 个CAD文件")
        
        # 预加载部分CAD特征（受限以避免内存峰值）
        if self.cache_cad:
            preload_count = min(len(cad_files), self.max_preload)
            print(f"预加载前 {preload_count} 个CAD模型特征 (最多 {self.max_preload})...")
            for cad_file in cad_files[:preload_count]:
                if cad_file not in self._cad_cache:
                    cad_pcd = self._load_stl(cad_file)
                    cad_prep = self._preprocess(cad_pcd)
                    cad_fpfh = self._compute_fpfh(cad_prep)
                    self._cad_cache[cad_file] = (cad_prep, cad_fpfh)
        
        all_results = []
        
        match_cnt = 0
        for scan_file in scan_files:
            print(f"\n{'='*50}")
            print(f"处理扫描文件: {Path(scan_file).name}")
            print(f"{'='*50}")
            
            # 与所有CAD匹配
            candidates = []
            for cad_file in cad_files:
                # print(f"\n匹配候选: {Path(cad_file).name}")
                try:
                    result = self.match_single(scan_file, cad_file)
                    candidates.append(result)
                except Exception as e:
                    print(f"  匹配失败: {e}")
                    continue
            
            if not candidates:
                print("警告: 未找到有效匹配")
                continue
            
            # 选择最佳匹配（综合fitness和RMSE）
            best_result = min(candidates, key=lambda x: x.inlier_rmse / (x.fitness + 1e-6))
            all_results.append({
                'scan_file': scan_file,
                'best_match': best_result,
                'all_candidates': sorted(candidates, key=lambda x: x.inlier_rmse)
            })
            
            if scan_file == best_result.cad_file:
                match_cnt+=1
            print(f"\n最佳匹配: {Path(best_result.cad_file).name}")
            print(f"  Fitness: {best_result.fitness:.4f}")
            print(f"  RMSE: {best_result.inlier_rmse:.4f}mm")
            print(f"  处理时间: {best_result.processing_time:.2f}s")
        
        print(f"\n总匹配完成: {len(all_results)} / {len(scan_files)}")
        
        # 保存结果
        if output_json:
            self._save_results(all_results, output_json)
            
        return all_results
    
    def _visualize_result(self, 
                         source: o3d.geometry.PointCloud,
                         target: o3d.geometry.PointCloud,
                         transform: np.ndarray,
                         result: MatchResult):
        """可视化匹配结果"""
        source_transformed = source.transform(transform)
        
        # 着色
        source_transformed.paint_uniform_color([1, 0.706, 0])  # 黄色：扫描数据
        target.paint_uniform_color([0, 0.651, 0.929])  # 蓝色：CAD
        
        # 创建窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"Match Result - RMSE: {result.inlier_rmse:.3f}mm")
        vis.add_geometry(source_transformed)
        vis.add_geometry(target)
        
        # 设置视角
        opt = vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = [0.1, 0.1, 0.1]
        
        vis.run()
        vis.destroy_window()
    
    def _save_results(self, results: List[dict], output_path: str):
        """保存匹配结果到JSON"""
        output = []
        for item in results:
            entry = {
                'scan_file': Path(item['scan_file']).name,
                'best_match': {
                    'cad_file': Path(item['best_match'].cad_file).name,
                    'fitness': float(item['best_match'].fitness),
                    'inlier_rmse': float(item['best_match'].inlier_rmse),
                    'transformation': item['best_match'].transformation.tolist(),
                    'processing_time': float(item['best_match'].processing_time)
                },
                'top_candidates': [
                    {
                        'cad_file': Path(c.cad_file).name,
                        'fitness': float(c.fitness),
                        'rmse': float(c.inlier_rmse)
                    }
                    for c in item['all_candidates'][:3]
                ]
            }
            output.append(entry)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存至: {output_path}")


def quick_match(scan_folder: str = "scan_stl", 
                cad_folder: str = "cad_stl",
                visualize: bool = False):
    """
    快速匹配入口函数
    针对牙科口扫数据优化默认参数
    """
    # 针对口扫数据优化参数（单位mm）
    matcher = HighPrecisionPointCloudMatcher(
        voxel_size=0.2,      # 0.05mm高精度
        normal_radius=0.6,    # 法向量估计半径
        feature_radius=1.5,   # FPFH特征半径
        icp_threshold=1.0,    # ICP收敛阈值
        n_samples=10
    )
    
    results = matcher.match_batch(
        scan_folder=scan_folder,
        cad_folder=cad_folder,
        output_json="match_results.json"
    )
    
    return results


# 使用示例
if __name__ == "__main__":
    # 确保目录存在
    os.makedirs("scan_stl", exist_ok=True)
    os.makedirs("cad_stl", exist_ok=True)
    
    print("高精度点云匹配系统")
    print("==================")
    print("请将扫描STL文件放入 scan_stl 文件夹")
    print("请将CAD STL文件放入 cad_stl 文件夹")
    print()
    
    # 检查文件
    scan_files = glob.glob("scan_stl/*.stl")
    cad_files = glob.glob("cad_stl/*.stl")
    
    if not scan_files or not cad_files:
        print(f"扫描文件: {len(scan_files)}个")
        print(f"CAD文件: {len(cad_files)}个")
        print("请确保两个文件夹都包含STL文件后重新运行")
    else:
        # 执行匹配
        results = quick_match(visualize=True)
        
        # 打印摘要
        print("\n" + "="*50)
        print("匹配完成摘要")
        print("="*50)
        for r in results:
            bm = r['best_match']
            print(f"{Path(r['scan_file']).name} -> {Path(bm.cad_file).name}")
            print(f"  RMSE: {bm.inlier_rmse:.3f}mm | Fitness: {bm.fitness:.2%}")
