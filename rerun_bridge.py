import rerun as rr
import rerun.blueprint as rrb

class QuadrupedForceLogger:
    def __init__(self, app_name="Go2_Simulation"):
        # 初始化 Rerun
        rr.init(app_name, spawn=True)
        
        # 为每条腿和高度定义独立的视图 (SpaceView)
        # 这样可以确保“各曲线一个图”
        force_views = [
            rrb.TimeSeriesView(origin=f"Forces/{leg}", name=f"{leg} Leg Force (N)")
            for leg in ['FL', 'FR', 'RL', 'RR']
        ]
        
        torque_views = [
            rrb.TimeSeriesView(origin=f"Torques/{leg}", name=f"{leg} Thigh Torque (Nm)")
            for leg in ['FL', 'FR', 'RL', 'RR']
        ]
        
        # 构造蓝图：力与力矩分为两个大的 Tabs (页面)
        blueprint = rrb.Blueprint(
            rrb.Tabs(
                rrb.Vertical(
                    rrb.Grid(*force_views), 
                    name="力 (Forces)"
                ),
                rrb.Vertical(
                    rrb.Grid(*torque_views), 
                    name="力矩 (Torques)"
                ),
                rrb.TimeSeriesView(origin="Body/Height", name="机身高度 (Height)"),
                name="Main View"
            ),
            rrb.SelectionPanel(expanded=False),
            rrb.TimePanel(expanded=True),
        )
        
        # 发送蓝图配置到 Rerun Viewer
        rr.send_blueprint(blueprint)
        print(f"[Rerun] 窗口布局已优化：力与力矩分页面显示，单曲线独立绘图并已标注。")

    def log_sensor_forces(self, time_s, sensor_forces):
        """
        sensor_forces: {'FL': float, 'FR': float, 'RL': float, 'RR': float}
        """
        rr.set_time_seconds("sim_time", time_s)
        # 将数据记录到对应的路径下，蓝图会自动捕获
        for leg, value in sensor_forces.items():
            rr.log(f"Forces/{leg}", rr.Scalars(value))

    def log_thigh_torques(self, time_s, torques):
        """
        torques: {'FL': float, 'FR': float, 'RL': float, 'RR': float}
        """
        rr.set_time_seconds("sim_time", time_s)
        for leg, value in torques.items():
            rr.log(f"Torques/{leg}", rr.Scalars(value))

    def log_body_height(self, time_s, height):
        rr.set_time_seconds("sim_time", time_s)
        rr.log("Body/Height", rr.Scalars(height))
