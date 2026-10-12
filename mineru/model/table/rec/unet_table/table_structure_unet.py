# Copyright (c) Opendatalab. All rights reserved.

from docvortex.image import contour_area, morphology, trace_contours, minimum_rectangle, resize_image, rotation_matrix, transform_points, warp_image
import copy
import math
from typing import Optional, Dict, Any, Tuple

import numpy as np

from .utils import OrtInferSession, resize_img
from .utils_table_line_rec import (
    get_table_line,
    final_adjust_lines,
    draw_lines,
    adjust_lines,
    _iter_connected_component_coords,
    min_area_rect_box_from_components,
)
from.utils_table_recover import (
    sorted_ocr_boxes,
    box_4_2_poly_to_box_4_1,
)


class TSRUnet:
    def __init__(self, config: Dict):
        self.K = 1000
        self.MK = 4000
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        self.inp_height = 1024
        self.inp_width = 1024

        self.session = OrtInferSession(config)

    def __call__(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        img_info = self.preprocess(img)
        pred = self.infer(img_info)
        polygons, rotated_polygons = self.postprocess(img, pred, **kwargs)
        if polygons.size == 0:
            return None, None
        polygons = polygons.reshape(polygons.shape[0], 4, 2)
        polygons[:, 3, :], polygons[:, 1, :] = (
            polygons[:, 1, :].copy(),
            polygons[:, 3, :].copy(),
        )
        rotated_polygons = rotated_polygons.reshape(rotated_polygons.shape[0], 4, 2)
        rotated_polygons[:, 3, :], rotated_polygons[:, 1, :] = (
            rotated_polygons[:, 1, :].copy(),
            rotated_polygons[:, 3, :].copy(),
        )
        _, idx = sorted_ocr_boxes(
            [box_4_2_poly_to_box_4_1(poly_box) for poly_box in rotated_polygons],
            threhold=0.4,
        )
        polygons = polygons[idx]
        rotated_polygons = rotated_polygons[idx]
        return polygons, rotated_polygons

    def preprocess(self, img) -> Dict[str, Any]:

        scale = (self.inp_height, self.inp_width)
        img, _, _ = resize_img(img, scale, True)
        img = img.copy().astype(np.float32)
        assert img.dtype != np.uint8
        mean = np.float64(self.mean.reshape(1, -1))
        stdinv = 1 / np.float64(self.std.reshape(1, -1))
        img[:] = img[:, :, ::-1]  # 原位交换通道，保留输入数组所有权。
        np.subtract(img, mean, out=img, casting="unsafe")  # 保留 double 中间计算再量化为 float32。
        np.multiply(img, stdinv, out=img, casting="unsafe")
        img = img.transpose(2, 0, 1)
        images = img[None, :]
        return {"img": images}

    def infer(self, input):
        result = self.session(input["img"][None, ...])[0][0]
        result = result[0].astype(np.uint8)
        return result

    def postprocess(self, img, pred, **kwargs):

        row = kwargs.get("row", 50) if kwargs else 50
        col = kwargs.get("col", 30) if kwargs else 30
        h_lines_threshold = kwargs.get("h_lines_threshold", 100) if kwargs else 100
        v_lines_threshold = kwargs.get("v_lines_threshold", 15) if kwargs else 15
        angle = kwargs.get("angle", 50) if kwargs else 50
        enhance_box_line = kwargs.get("enhance_box_line", True) if kwargs else True
        morph_close = (
            kwargs.get("morph_close", enhance_box_line) if kwargs else enhance_box_line
        )  # 是否进行闭合运算以找到更多小的框
        more_h_lines = (
            kwargs.get("more_h_lines", enhance_box_line) if kwargs else enhance_box_line
        )  # 是否调整以找到更多的横线
        more_v_lines = (
            kwargs.get("more_v_lines", enhance_box_line) if kwargs else enhance_box_line
        )  # 是否调整以找到更多的横线
        extend_line = (
            kwargs.get("extend_line", enhance_box_line) if kwargs else enhance_box_line
        )  # 是否进行线段延长使得端点连接
        # 是否进行旋转修正
        rotated_fix = kwargs.get("rotated_fix") if kwargs else True
        ori_shape = img.shape
        pred = np.uint8(pred)
        hpred = copy.deepcopy(pred)  # 横线
        vpred = copy.deepcopy(pred)  # 竖线
        whereh = np.where(hpred == 1)
        wherev = np.where(vpred == 2)
        hpred[wherev] = 0
        vpred[whereh] = 0

        hpred = resize_image(hpred, (ori_shape[1], ori_shape[0]))
        vpred = resize_image(vpred, (ori_shape[1], ori_shape[0]))

        h, w = pred.shape
        hors_k = int(math.sqrt(w) * 1.2)
        vert_k = int(math.sqrt(h) * 1.2)
        hkernel = (hors_k, 1)
        vkernel = (1, vert_k)
        vpred = morphology(vpred, vkernel, operation="close")  # 先膨胀后腐蚀的过程
        if morph_close:
            hpred = morphology(hpred, hkernel, operation="close")
        colboxes = get_table_line(vpred, axis=1, lineW=col)  # 竖线
        rowboxes = get_table_line(hpred, axis=0, lineW=row)  # 横线
        rboxes_row_, rboxes_col_ = [], []
        if more_h_lines:
            rboxes_row_ = adjust_lines(rowboxes, alph=h_lines_threshold, angle=angle)
        if more_v_lines:
            rboxes_col_ = adjust_lines(colboxes, alph=v_lines_threshold, angle=angle)
        rowboxes += rboxes_row_
        colboxes += rboxes_col_
        if extend_line:
            rowboxes, colboxes = final_adjust_lines(rowboxes, colboxes)
        line_img = np.zeros(img.shape[:2], dtype="uint8")
        line_img = draw_lines(line_img, rowboxes + colboxes, color=255, lineW=2)
        rotated_angle = self.cal_rotate_angle(line_img)
        if rotated_fix and abs(rotated_angle) > 0.3:
            rotated_line_img = self.rotate_image(line_img, rotated_angle)
            rotated_polygons = self.cal_region_boxes(rotated_line_img)
            polygons = self.unrotate_polygons(
                rotated_polygons, rotated_angle, line_img.shape
            )
        else:
            polygons = self.cal_region_boxes(line_img)
            rotated_polygons = polygons.copy()
        return polygons, rotated_polygons

    def cal_region_boxes(self, tmp):
        components = _iter_connected_component_coords(tmp < 255)  # 8连通区域标记
        ceilboxes = min_area_rect_box_from_components(
            components,
            False,
            tmp.shape[1],
            tmp.shape[0],
            filtersmall=True,
            adjust_box=False,
        )  # 最后一个参数改为False
        return np.array(ceilboxes)

    def cal_rotate_angle(self, tmp):
        # 计算最外侧的旋转框

        contours = trace_contours(tmp, external_only=True)
        if not contours:
            return 0
        largest_contour = max(contours, key=contour_area)
        rect = minimum_rectangle(largest_contour)
        # 计算旋转角度
        angle = rect[2]
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90
        return angle

    def rotate_image(self, image, angle):
        # 获取图像的中心点

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        # 计算旋转矩阵
        M = rotation_matrix(center, angle, 1.0)

        # 进行旋转
        rotated_image = warp_image(image, M, (w, h), interpolation='nearest', border='replicate')

        return rotated_image

    def unrotate_polygons(
        self, polygons: np.ndarray, angle: float, img_shape: tuple
    ) -> np.ndarray:
        # 将多边形旋转回原始位置

        (h, w) = img_shape
        center = (w // 2, h // 2)
        M_inv = rotation_matrix(center, -angle, 1.0)

        # 将 (N, 8) 转换为 (N, 4, 2)
        polygons_reshaped = polygons.reshape(-1, 4, 2)

        # 批量逆旋转
        unrotated_polygons = transform_points(polygons_reshaped, M_inv)

        # 将 (N, 4, 2) 转换回 (N, 8)
        unrotated_polygons = unrotated_polygons.reshape(-1, 8)

        return unrotated_polygons
