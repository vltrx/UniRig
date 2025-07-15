from dataclasses import dataclass
from typing import Tuple, Union, List, Dict
from numpy import ndarray
import numpy as np
from abc import ABC, abstractmethod
from scipy.spatial.transform import Rotation as R

from .spec import ConfigSpec
from .asset import Asset
from .utils import axis_angle_to_matrix

@dataclass(frozen=True)
class AugmentAffineConfig(ConfigSpec):
    # final normalization cube
    normalize_into: Tuple[float, float]

    # randomly scale coordinates with probability p
    random_scale_p: float
    
    # scale range (lower, upper)
    random_scale: Tuple[float, float]
    
    # randomly shift coordinates with probability p
    random_shift_p: float
    
    # shift range (lower, upper)
    random_shift: Tuple[float, float]
    
    @classmethod
    def parse(cls, config) -> Union['AugmentAffineConfig', None]:
        if config is None:
            return None
        cls.check_keys(config)
        return AugmentAffineConfig(
            normalize_into=config.normalize_into,
            random_scale_p=config.get('random_scale_p', 0.),
            random_scale=config.get('random_scale', [1., 1.]),
            random_shift_p=config.get('random_shift_p', 0.),
            random_shift=config.get('random_shift', [0., 0.]),
        )

@dataclass(frozen=True)
class AugmentConfig(ConfigSpec):
    '''
    Config to handle final easy augmentation of vertices, normals and bones before sampling.
    '''    
    augment_affine_config: Union[AugmentAffineConfig, None]
    
    @classmethod
    def parse(cls, config) -> 'AugmentConfig':
        cls.check_keys(config)
        return AugmentConfig(
            augment_affine_config=AugmentAffineConfig.parse(config.get('augment_affine_config', None)),
        )

class Augment(ABC):
    '''
    Abstract class for augmentation
    '''
    def __init__(self):
        pass
    
    @abstractmethod
    def transform(self, asset: Asset, **kwargs):
        pass

    @abstractmethod
    def inverse(self, asset: Asset):
        pass

class AugmentAffine(Augment):
    
    def __init__(self, config: AugmentAffineConfig):
        super().__init__()
        self.config = config

    def _apply(self, v: ndarray, trans: ndarray) -> ndarray:
        return np.matmul(v, trans[:3, :3].transpose()) + trans[:3, 3]

    def transform(self, asset: Asset, **kwargs):
        bound_min = asset.vertices.min(axis=0)
        bound_max = asset.vertices.max(axis=0)
        if asset.joints is not None:
            joints_bound_min = asset.joints.min(axis=0)
            joints_bound_max = asset.joints.max(axis=0)            
            bound_min = np.minimum(bound_min, joints_bound_min)
            bound_max = np.maximum(bound_max, joints_bound_max)
        
        trans_vertex = np.eye(4, dtype=np.float32)
        
        trans_vertex = _trans_to_m(-(bound_max + bound_min)/2) @ trans_vertex
        
        # scale into the cube
        normalize_into = self.config.normalize_into
        scale = np.max((bound_max - bound_min) / (normalize_into[1] - normalize_into[0]))
        trans_vertex = _scale_to_m(1. / scale) @ trans_vertex
        
        bias = (normalize_into[0] + normalize_into[1]) / 2
        trans_vertex = _trans_to_m(np.array([bias, bias, bias], dtype=np.float32)) @ trans_vertex
        
        if np.random.rand() < self.config.random_scale_p:
            scale = _scale_to_m(np.random.uniform(self.config.random_scale[0], self.config.random_scale[1]))
            trans_vertex = scale @ trans_vertex
            
        if np.random.rand() < self.config.random_shift_p:
            l, r = self.config.random_shift
            shift = _trans_to_m(np.array([np.random.uniform(l, r), np.random.uniform(l, r), np.random.uniform(l, r)]), dtype=np.float32)
            trans_vertex = shift @ trans_vertex
        
        asset.vertices = self._apply(asset.vertices, trans_vertex)
        # do not affect scale in matrix
        if asset.matrix_local is not None:
            asset.matrix_local[:, :, 3:4] = trans_vertex @ asset.matrix_local[:, :, 3:4]
        if asset.pose_matrix is not None:
            asset.pose_matrix[:, :, 3:4] = trans_vertex @ asset.pose_matrix[:, :, 3:4]
        # do not affect normal here
        if asset.joints is not None:
            asset.joints = self._apply(asset.joints, trans_vertex)
        if asset.tails is not None:
            asset.tails = self._apply(asset.tails, trans_vertex)
        
        self.trans_vertex = trans_vertex
    
    def inverse(self, asset: Asset):
        m = np.linalg.inv(self.trans_vertex)
        asset.vertices = self._apply(asset.vertices, m)
        if asset.joints is not None:
            asset.joints = self._apply(asset.joints, m)
        if asset.tails is not None:
            asset.tails = self._apply(asset.tails, m)

def _trans_to_m(v: ndarray):
    m = np.eye(4, dtype=np.float32)
    m[0:3, 3] = v
    return m

def _scale_to_m(r: ndarray):
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = r
    m[1, 1] = r
    m[2, 2] = r
    m[3, 3] = 1.
    return m

def get_augments(config: AugmentConfig) -> Tuple[List[Augment], List[Augment]]:
    first_augments  = [] # augments before sample
    second_augments = [] # augments after sample
    augment_affine_config           = config.augment_affine_config

    if augment_affine_config is not None:
        second_augments.append(AugmentAffine(config=augment_affine_config))
    return first_augments, second_augments