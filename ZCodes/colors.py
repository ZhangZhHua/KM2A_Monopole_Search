# plt 配色文件
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors

# 预定义颜色
COLORS = {
    "green":"#80BEAF",
    "purple": "#C8A8DA", 
    "blue":"#79CEED", 
    "orange":"#F5B994",
    "gray": "#7C8A97",
    "brown":"#C6C09C",
    "red": "#FD475D", 
    "yellow":"#FED154", 
    "pink":"#FFC4C8",
}

# 预定义颜色组合
PALETTES = {
    "shinkai": ["#92C7E9", "#5678A6", "#F4A261", "#FAD0C4", "#594157"],  # 新海诚
    "ghibli": ["#A8C66C", "#B85042", "#E7A977", "#3E6990", "#EAD7A1"],  # 宫崎骏
    "cyberpunk": ["#00A8E8", "#8A00E8", "#FF1F8F", "#00E88F", "#F7E300"],  # 赛博朋克
    "monet": ["#A8DADC", "#FFB5A7", "#CDB4DB", "#457B9D", "#FFD166"],  # 莫奈
    "nordic": ["#2E3440", "#88C0D0", "#8FBC8F", "#5E81AC", "#4C566A"],  # 北欧极光
    
    "default": ["#F4A7B9","#8FBC8F","#FFD700","#80BEAF","#FD475D"],  # 默认
    "light_colors": ["#E49AAB", "#F3BDD7", "#FFDFA2", "#BFE4FF", "#A3B5FD", "#C3C5F8"],  # 轻盈色系
    "ice_cream": ["#D6A3DC", "#F7DB70", "#EABEBF", "#75CCE8", "#A5DEE5"],  # 冰淇淋色系
    "summer_red": ["#FEA78C", "#FFA3A6", "#F583B4", "#CD69A7", "#ED7179"],  # 夏日红
    "summer_alive": ["#FD475D", "#FFB284", "#E79796", "#AED4D5", "#FFC98B", "#C6C09C", "#F5CEC7"],  # 夏日生机
    "summer_green": ["#879E46", "#BBD5A6", "#FBCEB9", "#FEBD3D", "#E57B87"],  # 夏日绿
    "summer_bright": ["#FFC4C8", "#FF5685", "#9BC768", "#BB9DCF", "#FED154", "#FEB25E", "#6AC7E6"],  # 夏日明亮
    "summer_ice": ["#60EFDB", "#BEF2E5", "#C5E7F1", "#79CEED", "#6F89A2"],  # 夏日冰感
    "summer_lemon": ["#85CBCD", "#A8DEE0", "#F9E2AE", "#FBC78D", "#A7D676"],  # 夏日柠檬
    "summer_pomelo": ["#ADC965", "#89D5C9", "#FAC172", "#FF8357", "#E25B45"],  # 夏日柚子
    "summer_colors": ["#80BEAF", "#C8A8DA", "#79CEED", "#F5B994", "#7C8A97","#FFC4C8","#C6C09C", "#FD475D", "#FED154",  ],  # 夏日缤纷
    "summer_purple": ["#8869A5", "#C58ADE", "#B1BEEA", "#90C4E9", "#8095CE"],  # 夏日紫
}
# 预定义 colormap（渐变）
COLORMAPS = {
    name: LinearSegmentedColormap.from_list(name, colors)
    for name, colors in PALETTES.items()
}

def get_color(name):
    """获取颜色"""
    if name in COLORS:
        return COLORS[name]
    else:
        raise ValueError(f"未知 color name: {name}, 可选: {list(COLORS.keys())}")
    
def set_palette(name='default'):
    """设置 matplotlib 的默认颜色循环"""
    if name in PALETTES:
        plt.rcParams["axes.prop_cycle"] = cycler(color=PALETTES[name])
    else:
        raise ValueError(f"未知配色方案: {name}, 可选: {list(PALETTES.keys())}")

def get_colormap(name):
    """获取 colormap 对象（用于渐变映射）"""
    if name in COLORMAPS:
        return COLORMAPS[name]
    else:
        raise ValueError(f"未知 colormap: {name}, 可选: {list(COLORMAPS.keys())}")

def HEX_RGB(HexCode):
    # HEX → RGB 归一化 (0-1)
    return mcolors.to_rgb(HexCode)  # (0.5725490196078431, 0.7803921568627451, 0.9137254901960784)
def RGB_HEX(RGBtuple):
    # RGB 归一化 (0-1) → HEX
    return mcolors.to_hex(RGBtuple)  # '#92c7e8'
def print_color(name):
    if name in COLORS:
        print(COLORS[name])
    else:
        raise ValueError(f"未知 color name: {name}, 可选: {list(COLORS.keys())}")
def print_palette(name):
    if name in PALETTES:
        print(PALETTES[name])
    else:
        raise ValueError(f"未知 palette name: {name}, 可选: {list(PALETTES.keys())}")

def show_color(name):
    if name in COLORS:
        """在一个独立图像中显示单个颜色块"""
        fig, ax = plt.subplots(figsize=(1, 1))  # 生成 1x1 的 figure
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=COLORS[name]))  # 添加矩形色块
        ax.set_xticks([])  # 隐藏坐标轴
        ax.set_yticks([])
        ax.set_frame_on(False)  # 隐藏边框
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.show()
        plt.close()
    else:
        raise ValueError(f"未知 color name: {name}, 可选: {list(COLORS.keys())}")

def show_palette(palette_name):
    """在一个图中展示整个配色方案"""
    if palette_name not in PALETTES:
        raise ValueError(f"未知配色: {palette_name}, 可选: {list(PALETTES.keys())}")
    
    colors = PALETTES[palette_name]
    num_colors = len(colors)

    fig, ax = plt.subplots(figsize=(num_colors, 1))  # 根据颜色数量调整宽度
    ax.set_xlim(0, num_colors)
    ax.set_ylim(0, 1)

    # 画色块
    for i, color in enumerate(colors):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))

    ax.set_xticks([])  # 不显示坐标轴
    ax.set_yticks([])
    ax.set_frame_on(False)  # 隐藏边框
    plt.show()
    plt.close()

def show_all_palettes():
    """展示所有配色方案，每行一个"""
    num_palettes = len(PALETTES)
    fig, axes = plt.subplots(num_palettes, 1, figsize=(6, num_palettes * 0.5))

    if num_palettes == 1:
        axes = [axes]  # 处理单个配色的情况

    for ax, (palette_name, colors) in zip(axes, PALETTES.items()):
        num_colors = len(colors)
        ax.set_xlim(0, num_colors)
        ax.set_ylim(0, 1)

        # 画色块
        for i, color in enumerate(colors):
            ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        ax.set_title(palette_name, fontsize=10, loc="left")

    plt.tight_layout()
    plt.show()
    plt.close()