import numpy as np
from scipy.ndimage import distance_transform_edt as edt
from scipy.ndimage import binary_dilation, generate_binary_structure
import math

class ESDFManager:
    def __init__(self, vertices,xy_offset=(0.0, 0.0)):
        # self.cfg = cfg
        # self.unknown_space = cfg.unknown_space
        # self.occupied_thresh = cfg.occupied_thresh
        # self.free_thresh = cfg.free_thresh
        # self.truncation_distance = cfg.truncation_distance
        self.esdf = None
        self.occ_map = None
        self.map_resolution = 0.05
        self.origin = np.zeros(2)
    
        self.map_width=None
        self.map_height= None
        self.vertices= np.asarray(vertices,dtype=np.float32).copy()
        self.offset = xy_offset
        self.z_threshold= 0.05  # 降低阈值，让更多地形点被包含
        self.dilate_pixels=3
        self.make_grid_from_vertices()
        self.build_occupancy_grid_map()
        self.build_esdf()

    def make_grid_from_vertices(self):
        """
        Compute the grid from the mesh vertices.
        """
        
        self.vertices[:,0]-=self.offset[0]
        self.vertices[:,1]-=self.offset[1]

        min_x,min_y = self.vertices[:,0].min(), self.vertices[:,1].min()
        max_x,max_y = self.vertices[:,0].max(), self.vertices[:,1].max()

        self.map_width = int(np.ceil((max_x-min_x)/self.map_resolution))
        self.map_height = int(np.ceil((max_y-min_y)/self.map_resolution))
        self.origin = np.array([min_x, min_y], dtype=np.float32)
         
    def build_occupancy_grid_map(self):

        self.occ_map = np.zeros((self.map_height, self.map_width), dtype=np.uint8)


        mask = self.vertices[:,2] > self.z_threshold
        if not np.any(mask):
            return self.occ_map  # 没有障碍点

        xs = self.vertices[mask, 0] 
        ys = self.vertices[mask, 1]

        is_ = np.floor((xs - self.origin[0])/self.map_resolution).astype(np.int32)
        js_ = np.floor((ys - self.origin[1])/self.map_resolution).astype(np.int32)

        valid = (is_>=0)&(is_<self.map_width)&(js_>=0)&(js_<self.map_height)
        self.occ_map[js_[valid], is_[valid]] = 1

        if self.dilate_pixels and self.dilate_pixels > 0:
            # 8 邻域结构元素
            st = generate_binary_structure(2, 2)
            self.occ_map = binary_dilation(self.occ_map, structure=st, iterations=int(self.dilate_pixels)).astype(np.uint8)

        return self.occ_map
    
    def build_esdf(self):

        occ = self.occ_map
        res = self.map_resolution

        # 自由处到障碍的距离（像素）-> 乘 res 变米
        d_free = edt(occ == 0).astype(np.float32) * res
        # 障碍内部到自由的距离
        d_occ  = edt(occ == 1).astype(np.float32) * res

        esdf = d_free.copy()
        esdf[occ == 1] = -d_occ[occ == 1]

        self.esdf = esdf

    def idx_to_world(self,i, j):
        ox, oy = self.origin
        res=self.map_resolution
        x = ox + (i + 0.5) * res
        y = oy + (j + 0.5) * res
        return x, y

    def world_to_idx(self,x, y):
        ox, oy = self.origin
        res=self.map_resolution
        i = int(np.floor((x - ox) / res))
        j = int(np.floor((y - oy) / res))
        return i, j
    def esdf_grid_centers(self,):
        """
        返回:
        X, Y: 形状 (H, W)，每个元素是对应格中心的世界坐标 (x,y)
        """
        ox, oy = self.origin
        res=self.map_resolution
        xs = ox + (np.arange(self.map_width, dtype=np.float32) + 0.5) * res  # 列方向
        ys = oy + (np.arange(self.map_height, dtype=np.float32) + 0.5) * res  # 行方向
        X, Y = np.meshgrid(xs, ys)  # 默认 indexing='xy'，得到形状 (H, W)
        return X, Y