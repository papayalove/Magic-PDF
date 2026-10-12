# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy

import numpy as np
from docvortex.image import perspective_matrix, warp_image
from loguru import logger
from numpy import sqrt
from PIL import Image, ImageDraw


def Homography(
    image: np.ndarray,
    img_points: np.ndarray,
    world_width: float,
    world_height: float,
    interpolation: str | int | None = None,
    ratio_width: float = 1.0,
    ratio_height: float = 1.0,
) -> np.ndarray:
    """按既有四点单应矩阵采样，保留插值、扩边和目标尺寸规则。"""
    if interpolation is None:
        interpolation = "cubic"

    if isinstance(interpolation, int):
        interpolation = {0: "nearest", 1: "linear", 2: "cubic", 3: "area", 4: "lanczos4"}[interpolation]
    _points = np.array(img_points).reshape(-1, 2).astype(np.float32)

    expand_x = int(0.5 * world_width * (ratio_width - 1))
    expand_y = int(0.5 * world_height * (ratio_height - 1))

    pt_lefttop = [expand_x, expand_y]
    pt_righttop = [expand_x + world_width, expand_y]
    pt_leftbottom = [expand_x + world_width, expand_y + world_height]
    pt_rightbottom = [expand_x, expand_y + world_height]

    pts_std = np.float32([pt_lefttop, pt_righttop, pt_leftbottom, pt_rightbottom])

    img_crop_width = int(world_width * ratio_width)
    img_crop_height = int(world_height * ratio_height)

    M = perspective_matrix(_points, pts_std)

    dst_img = warp_image(image, M, (img_crop_width, img_crop_height), border="constant", interpolation=interpolation)

    return dst_img


class CurveTextRectifier:
    def vertical_text_process(
        self, points: np.ndarray | list[float] | list[list[float]], org_size: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        """竖排点序转入原横排展开逻辑，再恢复单应变换使用的坐标。"""
        org_w, org_h = org_size
        _points = np.array(points).reshape(-1).tolist()
        _points = np.array(_points[2:] + _points[:2]).reshape(-1, 2)

        adjusted_points = np.zeros(_points.shape, dtype=np.float32)
        adjusted_points[:, 0] = _points[:, 1]
        adjusted_points[:, 1] = org_h - _points[:, 0] - 1

        _image_coord, _world_coord, _new_image_size = self.horizontal_text_process(adjusted_points)

        image_coord = _points.reshape(1, -1, 2)
        world_coord = np.zeros(_world_coord.shape, dtype=np.float32)
        world_coord[:, :, 0] = 0 - _world_coord[:, :, 1]
        world_coord[:, :, 1] = _world_coord[:, :, 0]
        world_coord[:, :, 2] = _world_coord[:, :, 2]
        new_image_size = (_new_image_size[1], _new_image_size[0])

        return image_coord, world_coord, new_image_size

    def horizontal_text_process(
        self, points: np.ndarray | list[float] | list[list[float]]
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        """沿上下边界的既有距离计算展开尺寸和对应平面坐标。"""
        poly = np.array(points).reshape(-1)

        dx_list = []
        dy_list = []
        for i in range(1, len(poly) // 2):
            xdx = poly[i * 2] - poly[(i - 1) * 2]
            xdy = poly[i * 2 + 1] - poly[(i - 1) * 2 + 1]
            d = sqrt(xdx**2 + xdy**2)
            dx_list.append(d)

        for i in range(0, len(poly) // 4):
            ydx = poly[i * 2] - poly[len(poly) - 1 - (i * 2 + 1)]
            ydy = poly[i * 2 + 1] - poly[len(poly) - 1 - (i * 2)]
            d = sqrt(ydx**2 + ydy**2)
            dy_list.append(d)

        dx_list = [(dx_list[i] + dx_list[len(dx_list) - 1 - i]) / 2 for i in range(len(dx_list) // 2)]

        height = np.around(np.mean(dy_list))

        rect_coord = [0, 0]
        for i in range(0, len(poly) // 4 - 1):
            x = rect_coord[-2]
            x += dx_list[i]
            y = 0
            rect_coord.append(x)
            rect_coord.append(y)

        rect_coord_half = copy.deepcopy(rect_coord)
        for i in range(0, len(poly) // 4):
            x = rect_coord_half[len(rect_coord_half) - 2 * i - 2]
            y = height
            rect_coord.append(x)
            rect_coord.append(y)

        np_rect_coord = np.array(rect_coord).reshape(-1, 2)
        x_min = np.min(np_rect_coord[:, 0])
        y_min = np.min(np_rect_coord[:, 1])
        x_max = np.max(np_rect_coord[:, 0])
        y_max = np.max(np_rect_coord[:, 1])
        new_image_size = (int(x_max - x_min + 0.5), int(y_max - y_min + 0.5))
        x_mean = (x_max - x_min) / 2
        y_mean = (y_max - y_min) / 2
        np_rect_coord[:, 0] -= x_mean
        np_rect_coord[:, 1] -= y_mean
        rect_coord = np_rect_coord.reshape(-1).tolist()

        rect_coord = np.array(rect_coord).reshape(-1, 2)
        world_coord = np.ones((len(rect_coord), 3)) * 0

        world_coord[:, :2] = rect_coord

        image_coord = np.array(poly).reshape(1, -1, 2)
        world_coord = world_coord.reshape(1, -1, 3)

        return image_coord, world_coord, new_image_size

    def horizontal_text_estimate(self, points: np.ndarray | list[float] | list[list[float]]) -> bool:
        """沿用原外接框宽高比判定横竖方向。"""
        pts = np.array(points).reshape(-1, 2)
        x_min = int(np.min(pts[:, 0]))
        y_min = int(np.min(pts[:, 1]))
        x_max = int(np.max(pts[:, 0]))
        y_max = int(np.max(pts[:, 1]))
        x = x_max - x_min
        y = y_max - y_min
        is_horizontal_text = True
        if y / x > 1.5:
            is_horizontal_text = False
        return is_horizontal_text

    def dc_homo(
        self,
        img: np.ndarray,
        img_points: np.ndarray,
        obj_points: np.ndarray,
        is_horizontal_text: bool,
        interpolation: str | int | None = None,
        ratio_width: float = 1.0,
        ratio_height: float = 1.0,
    ) -> np.ndarray:
        """逐段单应展开后按原高度拼接，竖排结果保留原旋转方向。"""
        if interpolation is None:
            interpolation = "linear"

        _img_points = img_points.reshape(-1, 2)
        _obj_points = obj_points.reshape(-1, 3)

        homo_img_list = []
        width_list = []
        height_list = []
        for i in range(len(_img_points) // 2 - 1):
            new_img_points = np.zeros((4, 2)).astype(np.float32)
            new_obj_points = np.zeros((4, 2)).astype(np.float32)

            new_img_points[0:2, :] = _img_points[i : (i + 2), :2]
            new_img_points[2:4, :] = _img_points[::-1, :][i : (i + 2), :2][::-1, :]

            new_obj_points[0:2, :] = _obj_points[i : (i + 2), :2]
            new_obj_points[2:4, :] = _obj_points[::-1, :][i : (i + 2), :2][::-1, :]

            if is_horizontal_text:
                world_width = np.abs(new_obj_points[1, 0] - new_obj_points[0, 0])
                world_height = np.abs(new_obj_points[3, 1] - new_obj_points[0, 1])
            else:
                world_width = np.abs(new_obj_points[1, 1] - new_obj_points[0, 1])
                world_height = np.abs(new_obj_points[3, 0] - new_obj_points[0, 0])

            homo_img = Homography(
                img,
                new_img_points,
                world_width,
                world_height,
                interpolation=interpolation,
                ratio_width=ratio_width,
                ratio_height=ratio_height,
            )

            homo_img_list.append(homo_img)
            _h, _w = homo_img.shape[:2]
            width_list.append(_w)
            height_list.append(_h)

        rectified_image = np.zeros((np.max(height_list), sum(width_list), 3)).astype(np.uint8)

        st = 0
        for homo_img, w, h in zip(homo_img_list, width_list, height_list):
            rectified_image[:h, st : st + w, :] = homo_img
            st += w

        if not is_horizontal_text:
            rectified_image = np.rot90(rectified_image, 3)

        return rectified_image

    def __call__(
        self,
        image_data: np.ndarray,
        points: np.ndarray | list[float] | list[list[float]],
        interpolation: str | int | None = None,
        ratio_width: float = 1.0,
        ratio_height: float = 1.0,
        mode: str = "homography",
    ) -> tuple[np.ndarray, float]:
        """只执行既有分段单应变换，旧相机标定模式明确拒绝。"""
        if mode.lower() != "homography":
            raise ValueError(f'Only mode="homography" is supported, got {mode!r}')
        if interpolation is None:
            interpolation = "linear"
        org_h, org_w = image_data.shape[:2]
        is_horizontal_text = self.horizontal_text_estimate(points)
        if is_horizontal_text:
            image_coord, world_coord, _ = self.horizontal_text_process(points)
        else:
            image_coord, world_coord, _ = self.vertical_text_process(points, (org_w, org_h))
        dst = self.dc_homo(
            image_data,
            image_coord,
            world_coord,
            is_horizontal_text,
            interpolation=interpolation,
            ratio_width=1.0,
            ratio_height=1.0,
        )
        return dst, 0.01


class AutoRectifier:
    def __init__(self) -> None:
        """保留曲线文字点数阈值，初始化不再创建虚拟相机。"""
        self.npoints = 10

    @staticmethod
    def get_rotate_crop_image(
        img: np.ndarray,
        points: np.ndarray | list[float] | list[list[float]],
        interpolation: str | int | None = None,
        ratio_width: float = 1.0,
        ratio_height: float = 1.0,
    ) -> np.ndarray:
        """四点使用既有透视裁图，其他点数保留独立外接矩形裁片。"""
        if interpolation is None:
            interpolation = "cubic"
        h, w = img.shape[:2]
        _points = np.array(points).reshape(-1, 2).astype(np.float32)

        if len(_points) != 4:
            x_min = int(np.min(_points[:, 0]))
            y_min = int(np.min(_points[:, 1]))
            x_max = int(np.max(_points[:, 0]))
            y_max = int(np.max(_points[:, 1]))
            dx = x_max - x_min
            dy = y_max - y_min
            expand_x = int(0.5 * dx * (ratio_width - 1))
            expand_y = int(0.5 * dy * (ratio_height - 1))
            x_min = np.clip(int(x_min - expand_x), 0, w - 1)
            y_min = np.clip(int(y_min - expand_y), 0, h - 1)
            x_max = np.clip(int(x_max + expand_x), 0, w - 1)
            y_max = np.clip(int(y_max + expand_y), 0, h - 1)

            dst_img = img[y_min:y_max, x_min:x_max, :].copy()
        else:
            img_crop_width = int(
                max(
                    np.linalg.norm(_points[0] - _points[1]),
                    np.linalg.norm(_points[2] - _points[3]),
                )
            )
            img_crop_height = int(
                max(
                    np.linalg.norm(_points[0] - _points[3]),
                    np.linalg.norm(_points[1] - _points[2]),
                )
            )

            dst_img = Homography(
                img,
                _points,
                img_crop_width,
                img_crop_height,
                interpolation,
                ratio_width,
                ratio_height,
            )

        return dst_img

    def visualize(self, image_data: np.ndarray, points_list: list[list[float] | list[list[float]]]) -> np.ndarray:
        """使用 Pillow 标注印章矫正点，返回与输入同通道顺序的诊断图。"""
        with Image.fromarray(image_data[:, :, ::-1]) as canvas:
            draw = ImageDraw.Draw(canvas)
            for box in points_list:
                points = [tuple(map(int, xy)) for xy in np.asarray(box).reshape(-1, 2)]
                draw.line(points + points[:1], fill=(255, 0, 0), width=2)
                for index, (x, y) in enumerate(points):
                    draw.ellipse((x - 1, y - 1, x + 1, y + 1), outline=(0, 255, 255) if index == 0 else (0, 0, 255), width=2)
            return np.asarray(canvas)[:, :, ::-1].copy()

    def __call__(
        self,
        image_data: np.ndarray,
        points: np.ndarray | list[float] | list[list[float]],
        interpolation: str | int | None = None,
        ratio_width: float = 1.0,
        ratio_height: float = 1.0,
        mode: str = "homography",
    ) -> np.ndarray:
        """默认使用曲线单应矫正，几何异常保留原有外接矩形裁图回退。"""
        if mode.lower() != "homography":
            raise ValueError(f'Only mode="homography" is supported, got {mode!r}')
        if interpolation is None:
            interpolation = "linear"
        _points = np.array(points).reshape(-1, 2)
        if len(_points) >= self.npoints and len(_points) % 2 == 0:
            try:
                curve_text_rectifier = CurveTextRectifier()
                dst_img, _ = curve_text_rectifier(image_data, points, interpolation, ratio_width, ratio_height, mode)
            except Exception as e:
                logger.warning(f"Exception caught: {e}")
                dst_img = self.get_rotate_crop_image(image_data, points, interpolation, ratio_width, ratio_height)
        else:
            dst_img = self.get_rotate_crop_image(image_data, _points, interpolation, ratio_width, ratio_height)
        return dst_img

    def run(
        self,
        image_data: np.ndarray,
        points_list: list[list[float] | list[list[float]]],
        interpolation: str | int | None = None,
        ratio_width: float = 1.0,
        ratio_height: float = 1.0,
        mode: str = "homography",
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """批量单应矫正并返回诊断图，不再提供相机标定或损失阈值参数。"""
        if image_data is None:
            raise ValueError
        if not isinstance(points_list, list):
            raise ValueError
        for points in points_list:
            if not isinstance(points, list):
                raise ValueError
        if interpolation is None:
            interpolation = "linear"
        if ratio_width < 1.0 or ratio_height < 1.0:
            raise ValueError(
                "ratio_width and ratio_height cannot be smaller than 1, but got {}",
                (ratio_width, ratio_height),
            )
        if mode.lower() != "homography":
            raise ValueError(f'Only mode="homography" is supported, got {mode!r}')
        if mode.lower() == "homography" and ratio_width != 1.0 and ratio_height != 1.0:
            raise ValueError(
                "ratio_width and ratio_height must be 1.0 when mode is homography, but got mode:{}, ratio:({},{})".format(
                    mode, ratio_width, ratio_height
                )
            )
        res = []
        for points in points_list:
            rectified_img = self(
                image_data,
                points,
                interpolation,
                ratio_width,
                ratio_height,
                mode=mode,
            )
            res.append(rectified_img)
        visualized_image = self.visualize(image_data, points_list)
        return res, visualized_image
