import numpy as np
from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import array_to_string, new_body, new_geom, string_to_array, xml_path_completion

class ShelfArena(Arena):
    """
    Workspace that contains a shelf.
    仿照 TableArena 编写，直接传递 XML 路径给父类。
    
    Args:
        shelf_pos (3-tuple): (x,y,z) offset for the shelf position.
        xml (str): xml file to load arena.
    """

    def __init__(
        self,
        shelf_pos=(0, 0, 0),
        support_platform_pos=(0.0, 0.09, 0.04),
        support_platform_size=(0.58, 0.40, 0.5),
        xml="arenas/shelf_arena.xml",
    ):
        # 1. 【关键修复】直接传递 xml 路径作为位置参数，不要加 xml_file=
        super().__init__(xml_path_completion(xml))

        self.shelf_pos = np.array(shelf_pos)
        self.support_platform_pos = np.array(support_platform_pos)
        self.support_platform_size = np.array(support_platform_size)
        self.support_platform_half_size = self.support_platform_size / 2.0
        
        # 2. 获取货架相关的 body 和 geom (仿照 TableArena 查找 table 的逻辑)
        # 假设你的 shelf_arena.xml 中有一个名为 "shelf" 的 body
        # 如果 XML 中名字不同，请修改这里的 name 属性
        self.shelf_body = self.worldbody.find("./body[@name='shelf']")
        
        if self.shelf_body is None:
            # 尝试查找其他可能的名字，或者抛出警告
            # 有些 XML 可能直接把 geom 放在 worldbody 下，没有包裹在 body 里
            print("[ShelfArena] 警告：未在 XML 中找到名为 'shelf' 的 body。尝试直接使用 worldbody。")
            self.shelf_body = self.worldbody
        else:
            # 如果找到了 body，应用偏移
            self._configure_shelf_location()

        self._add_support_platform()

    def _configure_shelf_location(self):
        """
        配置货架的位置
        """
        # 如果 XML 中定义了 shelf 的初始位置，这里可以进行微调
        # 例如：self.shelf_body.set("pos", array_to_string(self.shelf_pos))
        
        # 注意：如果 shelf_pos 是相对于 arena 中心的偏移，你可能需要结合 floor 的位置计算
        # 这里简单示例：直接设置 body 的 pos 属性
        if np.any(self.shelf_pos != 0):
            current_pos_str = self.shelf_body.get("pos")
            if current_pos_str:
                current_pos = string_to_array(current_pos_str)
                new_pos = current_pos + self.shelf_pos
                self.shelf_body.set("pos", array_to_string(new_pos))
            else:
                self.shelf_body.set("pos", array_to_string(self.shelf_pos))

    def _add_support_platform(self):
        """
        在货架前方地面上添加一个低矮托架，方便夹爪抓取上面的物块。
        """
        platform_body = new_body(name="support_platform", pos=array_to_string(self.support_platform_pos))
        platform_collision = new_geom(
            name="support_platform_collision",
            type="box",
            size=array_to_string(self.support_platform_half_size),
            friction=array_to_string([1.0, 0.005, 0.0001]),
            rgba=array_to_string([0.55, 0.45, 0.30, 1.0]),
            group=0,
        )
        platform_visual = new_geom(
            name="support_platform_visual",
            type="box",
            size=array_to_string(self.support_platform_half_size),
            conaffinity="0",
            contype="0",
            rgba=array_to_string([0.65, 0.55, 0.38, 1.0]),
            group=1,
        )
        platform_body.append(platform_collision)
        platform_body.append(platform_visual)
        self.worldbody.append(platform_body)

    @property
    def support_platform_top_abs(self):
        """
        获取托架顶面的世界坐标高度。
        """
        return self.support_platform_pos + np.array([0.0, 0.0, self.support_platform_half_size[2]])

    @property
    def shelf_top_abs(self):
        """
        获取货架顶部的绝对位置 (示例属性，可根据需要修改)
        """
        # 这里需要根据你的 XML 结构具体实现
        # 暂时返回 shelf_pos 作为占位
        return self.shelf_pos
