# map_processor.py

import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import gaussian_filter, zoom
import rasterio


def process_tif_map(tif_filepath, target_display_size=2000):
    """
    读取并处理TIF高程图文件。

    参数:
    - tif_filepath (str): .tif 文件的路径。
    - target_display_size (int): 希望输出的地图数据的正方形尺寸。

    返回:
    - tuple or None: 如果成功，返回包含以下元素的元组：
        - final_data_for_plot (np.array): 处理后的最终地图数据。
        - cmap (ListedColormap): 自定义颜色图。
        - norm (BoundaryNorm): 颜色的归一化规则。
        - boundaries (list): 海拔边界。
        - display_width (int): 地图宽度。
        - display_height (int): 地图高度。
      如果失败，返回 None。
    """
    try:
        with rasterio.open(tif_filepath) as src:
            original_data = src.read(1)
            nodata_value = src.nodata
            src_height, src_width = src.shape
            print(f"读取 TIF 文件成功: {tif_filepath}")
            print(f"原始数据尺寸: {src_width}x{src_height}")
    except Exception as e:
        print(f"读取 TIF 文件时发生错误: {e}")
        return None

    # --- 数据预处理和缩放 ---
    display_height, display_width = target_display_size, target_display_size
    processed_data = original_data.astype(float)
    if nodata_value is not None:
        processed_data[original_data == nodata_value] = np.nan

    valid_data = processed_data[~np.isnan(processed_data)]
    if valid_data.size == 0:
        print("错误: TIF 文件中没有有效数据。")
        return None

    min_target_val, max_target_val = 0.00, 84
    data_scaled = np.full(processed_data.shape, np.nan)
    data_scaled[~np.isnan(processed_data)] = np.interp(
        valid_data, (valid_data.min(), valid_data.max()), (min_target_val, max_target_val)
    )

    if src_height != display_height or src_width != display_width:
        zoom_factor_y = display_height / src_height
        zoom_factor_x = display_width / src_width
        data_resampled = zoom(data_scaled, (zoom_factor_y, zoom_factor_x), order=1, mode='nearest')
        data_resampled[np.isnan(data_resampled)] = min_target_val
        final_data_for_plot = data_resampled
    else:
        final_data_for_plot = data_scaled

    final_data_for_plot = gaussian_filter(final_data_for_plot, sigma=11)

    # --- 定义自定义颜色图和归一化 ---
    colors = ['#76AEC2', '#B4CCA8', '#C9D8C0', '#F5F3EF', '#EAE8E3', '#B8AFA5', '#9D9288']
    cmap = ListedColormap(colors)
    boundaries = [0.00, 12, 24, 36, 48, 60, 72, 84]
    norm = BoundaryNorm(boundaries, cmap.N)

    return final_data_for_plot, cmap, norm, boundaries, display_width, display_height