#!/usr/bin/env python3

import time
import threading
import yaml
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from std_msgs.msg import String
from std_srvs.srv import Trigger
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from irobot_create_msgs.action import Dock


class MissionState(Enum):
    IDLE = 'IDLE'
    INITIALIZING = 'INITIALIZING'
    READY_FOR_DELIVERY = 'READY_FOR_DELIVERY'
    MISSION_STARTED = 'MISSION_STARTED'
    SETTING_INITIAL_POSE = 'SETTING_INITIAL_POSE'
    GOING_TO_PICKUP = 'GOING_TO_PICKUP'
    ALIGNING_PICKUP = 'ALIGNING_PICKUP'
    WAITING_UR_LOAD = 'WAITING_UR_LOAD'
    GOING_TO_DELIVERY = 'GOING_TO_DELIVERY'
    ALIGNING_DELIVERY = 'ALIGNING_DELIVERY'
    DELIVERING_ITEM = 'DELIVERING_ITEM'
    WAITING_AT_DELIVERY = 'WAITING_AT_DELIVERY'
    RETURNING_TO_PICKUP = 'RETURNING_TO_PICKUP'
    WAITING_UR_UNLOAD = 'WAITING_UR_UNLOAD'
    RETURNING_TO_CHARGER = 'RETURNING_TO_CHARGER'
    AUTO_DOCKING = 'AUTO_DOCKING'
    MISSION_COMPLETE = 'MISSION_COMPLETE'
    STOP_REQUESTED = 'STOP_REQUESTED'
    MISSION_CANCELLED = 'MISSION_CANCELLED'
    MISSION_FAILED = 'MISSION_FAILED'
    BUSY = 'BUSY'
    UNKNOWN_COMMAND = 'UNKNOWN_COMMAND'
    UNKNOWN_POINT = 'UNKNOWN_POINT'


class DeliveryMissionController(Node):
    def __init__(self):
        super().__init__('delivery_mission_controller')

        self.callback_group = ReentrantCallbackGroup()

        self.status_pub = self.create_publisher(String, '/delivery_status', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.initialpose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

        self.command_sub = self.create_subscription(
            String, '/delivery_command', self.command_callback, 10,
            callback_group=self.callback_group
        )

        self.task_list_sub = self.create_subscription(
            String, '/delivery_task_list', self.task_list_callback, 10,
            callback_group=self.callback_group
        )

        self.dynamic_goal_sub = self.create_subscription(
            PoseStamped, '/delivery_dynamic_goal', self.dynamic_goal_callback, 10,
            callback_group=self.callback_group
        )

        self.reload_srv = self.create_service(
            Trigger, '/reload_delivery_points', self.reload_points_callback,
            callback_group=self.callback_group
        )

        self.nav_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose',
            callback_group=self.callback_group
        )

        self.dock_client = ActionClient(
            self, Dock, 'dock',
            callback_group=self.callback_group
        )

        self.pickup_client = self.create_client(
            Trigger, '/pickup_item',
            callback_group=self.callback_group
        )

        self.unload_client = self.create_client(
            Trigger, '/unload_item',
            callback_group=self.callback_group
        )

        self.config_path = '/root/delivery_robot_project/config/delivery_points.yaml'
        self.points = {}
        self.load_points()

        self.delivery_wait_time = 180.0
        self.short_delivery_display_time = 2.0

        self.current_goal_handle = None
        self.mission_running = False
        self.cancel_requested = False
        self.mission_lock = threading.Lock()

        self.waiting_for_delivery_command = False
        self.pending_order = None
        self.order_event = threading.Event()

        self.pending_task_list = []
        self.pending_dynamic_goal = None

        self.current_state = MissionState.INITIALIZING

        self.publish_state(MissionState.INITIALIZING)

        self.get_logger().info('Waiting for Nav2 action server...')
        while not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn('Still waiting for Nav2 action server...')

        self.get_logger().info('Waiting for dock action server...')
        while not self.dock_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn('Still waiting for dock action server...')

        self.get_logger().info('All action servers available.')
        self.publish_state(MissionState.IDLE)

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------
    def publish_state(self, state: MissionState, extra: str = None):
        self.current_state = state
        text = state.value if extra is None else f'{state.value}:{extra}'
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)
        self.get_logger().info(f'[STATE] {text}')

    def load_points(self):
        try:
            with open(self.config_path, 'r') as f:
                self.points = yaml.safe_load(f) or {}
            self.get_logger().info(f'Loaded points: {list(self.points.keys())}')
            return True, f'Loaded {len(self.points)} points.'
        except Exception as e:
            self.points = {}
            self.get_logger().error(f'Failed to load points: {e}')
            return False, str(e)

    def reload_points_callback(self, request, response):
        ok, message = self.load_points()
        response.success = ok
        response.message = message
        return response

    def stop_robot(self):
        self.cmd_vel_pub.publish(Twist())

    def wait_for_future(self, future, timeout_sec=None):
        start_time = time.time()
        while not future.done():
            if self.cancel_requested:
                return None
            if timeout_sec is not None and (time.time() - start_time) > timeout_sec:
                return None
            time.sleep(0.1)
        return future.result()

    def pose_from_point(self, point: dict):
        pose = PoseStamped()
        pose.header.frame_id = point.get('frame_id', 'map')
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = float(point['x'])
        pose.pose.position.y = float(point['y'])
        pose.pose.position.z = float(point.get('z', 0.0))

        pose.pose.orientation.x = float(point.get('qx', 0.0))
        pose.pose.orientation.y = float(point.get('qy', 0.0))
        pose.pose.orientation.z = float(point.get('qz', 0.0))
        pose.pose.orientation.w = float(point.get('qw', 1.0))
        return pose

    def build_initialpose_msg(self, point: dict):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = point.get('frame_id', 'map')
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.pose.pose.position.x = float(point['x'])
        msg.pose.pose.position.y = float(point['y'])
        msg.pose.pose.position.z = float(point.get('z', 0.0))

        msg.pose.pose.orientation.x = float(point.get('qx', 0.0))
        msg.pose.pose.orientation.y = float(point.get('qy', 0.0))
        msg.pose.pose.orientation.z = float(point.get('qz', 0.0))
        msg.pose.pose.orientation.w = float(point.get('qw', 1.0))

        msg.pose.covariance[0] = 0.25
        msg.pose.covariance[7] = 0.25
        msg.pose.covariance[35] = 0.10
        return msg

    # --------------------------------------------------
    # Input callbacks
    # --------------------------------------------------
    def task_list_callback(self, msg: String):
        raw = msg.data.strip()
        if not raw:
            self.pending_task_list = []
            self.publish_state(MissionState.IDLE, 'TASK_LIST_EMPTY')
            return

        self.pending_task_list = [x.strip() for x in raw.split(',') if x.strip()]
        self.get_logger().info(f'Received task list: {self.pending_task_list}')
        self.publish_state(MissionState.IDLE, 'TASK_LIST_RECEIVED')

    def dynamic_goal_callback(self, msg: PoseStamped):
        self.pending_dynamic_goal = msg
        self.get_logger().info(
            f'Received dynamic goal: x={msg.pose.position.x:.3f}, y={msg.pose.position.y:.3f}'
        )
        self.publish_state(MissionState.IDLE, 'DYNAMIC_GOAL_RECEIVED')

    def command_callback(self, msg: String):
        command = msg.data.strip().lower()
        self.get_logger().info(f'Received command: {command}')

        if command == 'stop':
            self.stop_current_mission()
            return

        if self.waiting_for_delivery_command:
            if command == 'start_delivery':
                self.pending_order = 'start_delivery'
                self.waiting_for_delivery_command = False
                self.order_event.set()
                self.publish_state(MissionState.WAITING_AT_DELIVERY, 'ORDER_RECEIVED_START_DELIVERY')
            elif command == 'return_items':
                self.pending_order = 'return_items'
                self.waiting_for_delivery_command = False
                self.order_event.set()
                self.publish_state(MissionState.WAITING_AT_DELIVERY, 'ORDER_RECEIVED_RETURN_ITEMS')
            elif command == 'go_charge':
                self.pending_order = 'go_charge'
                self.waiting_for_delivery_command = False
                self.order_event.set()
                self.publish_state(MissionState.WAITING_AT_DELIVERY, 'ORDER_RECEIVED_GO_CHARGE')
            else:
                self.publish_state(MissionState.WAITING_AT_DELIVERY)
            return

        if self.mission_running:
            self.publish_state(MissionState.BUSY)
            return

        if command == 'start_delivery':
            threading.Thread(target=self.run_default_delivery, daemon=True).start()
        elif command == 'dock_robot':
            threading.Thread(target=self.run_dock_only, daemon=True).start()
        elif command == 'undock_robot':
            threading.Thread(target=self.run_prepare_only, daemon=True).start()
        elif command == 'go_charge':
            threading.Thread(target=self.run_go_charge_only, daemon=True).start()
        elif command.startswith('go_to:'):
            point_name = command.split(':', 1)[1].strip()
            threading.Thread(target=self.run_go_to_named_point, args=(point_name,), daemon=True).start()
        elif command == 'run_task_list':
            threading.Thread(target=self.run_task_list_mission, daemon=True).start()
        elif command == 'run_dynamic_goal':
            threading.Thread(target=self.run_dynamic_goal_mission, daemon=True).start()
        else:
            self.publish_state(MissionState.UNKNOWN_COMMAND)

    def stop_current_mission(self):
        self.cancel_requested = True
        self.waiting_for_delivery_command = False
        self.order_event.set()
        self.publish_state(MissionState.STOP_REQUESTED)

        if self.current_goal_handle is not None:
            self.current_goal_handle.cancel_goal_async()

        self.stop_robot()
        time.sleep(1.0)

    # --------------------------------------------------
    # Navigation
    # --------------------------------------------------
    def navigate_to_pose_stamped(self, pose: PoseStamped, label='dynamic_goal') -> bool:
        if self.cancel_requested:
            self.publish_state(MissionState.MISSION_CANCELLED)
            return False

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        send_goal_future = self.nav_client.send_goal_async(goal_msg)
        goal_handle = self.wait_for_future(send_goal_future, timeout_sec=20.0)

        if goal_handle is None:
            self.publish_state(
                MissionState.MISSION_CANCELLED if self.cancel_requested else MissionState.MISSION_FAILED
            )
            return False

        self.current_goal_handle = goal_handle

        if not goal_handle.accepted:
            self.current_goal_handle = None
            self.publish_state(MissionState.MISSION_FAILED, f'GOAL_REJECTED_{label.upper()}')
            return False

        result_future = goal_handle.get_result_async()
        result = self.wait_for_future(result_future, timeout_sec=600.0)
        self.current_goal_handle = None

        if result is None:
            self.publish_state(
                MissionState.MISSION_CANCELLED if self.cancel_requested else MissionState.MISSION_FAILED
            )
            return False

        if result.status == GoalStatus.STATUS_SUCCEEDED:
            self.stop_robot()
            time.sleep(1.0)
            return True

        if result.status == GoalStatus.STATUS_CANCELED:
            self.publish_state(MissionState.MISSION_CANCELLED)
            return False

        self.publish_state(MissionState.MISSION_FAILED, f'NAV_FAILED_{label.upper()}')
        return False

    def navigate_to_named_point(self, point_name: str) -> bool:
        if point_name not in self.points:
            self.publish_state(MissionState.UNKNOWN_POINT, point_name)
            return False

        if self.cancel_requested:
            self.publish_state(MissionState.MISSION_CANCELLED)
            return False

        point = self.points[point_name]
        self.get_logger().info(
            f"Sending goal to {point_name}: x={point['x']:.3f}, y={point['y']:.3f}"
        )

        return self.navigate_to_pose_stamped(self.pose_from_point(point), point_name)

    # --------------------------------------------------
    # Initial pose / dock
    # --------------------------------------------------
    def set_initial_pose_auto(self) -> bool:
        self.publish_state(MissionState.SETTING_INITIAL_POSE)

        if 'startup_pose' not in self.points:
            self.publish_state(MissionState.MISSION_FAILED, 'NO_STARTUP_POSE')
            return False

        msg = self.build_initialpose_msg(self.points['startup_pose'])

        for i in range(5):
            if self.cancel_requested:
                self.publish_state(MissionState.MISSION_CANCELLED)
                return False
            msg.header.stamp = self.get_clock().now().to_msg()
            self.initialpose_pub.publish(msg)
            self.get_logger().info(f'Published startup initial pose ({i+1}/5).')
            time.sleep(0.4)

        time.sleep(2.0)
        return True

    def auto_dock_to_charger(self) -> bool:
        self.publish_state(MissionState.AUTO_DOCKING)

        goal_msg = Dock.Goal()
        send_goal_future = self.dock_client.send_goal_async(goal_msg)
        goal_handle = self.wait_for_future(send_goal_future, timeout_sec=20.0)

        if goal_handle is None:
            self.publish_state(MissionState.MISSION_FAILED, 'DOCK_TIMEOUT')
            return False

        if not goal_handle.accepted:
            self.publish_state(MissionState.MISSION_FAILED, 'DOCK_REJECTED')
            return False

        result_future = goal_handle.get_result_async()
        result = self.wait_for_future(result_future, timeout_sec=180.0)

        if result is None:
            self.publish_state(MissionState.MISSION_FAILED, 'DOCK_FAILED')
            return False

        if result.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Dock succeeded.')
            time.sleep(1.0)
            return True

        self.publish_state(MissionState.MISSION_FAILED, f'DOCK_FAILED_STATUS_{result.status}')
        return False

    # --------------------------------------------------
    # Placeholder hooks for next phase
    # --------------------------------------------------
    def apriltag_align_pickup(self) -> bool:
        self.publish_state(MissionState.ALIGNING_PICKUP)
        self.get_logger().warn('apriltag_align_pickup() not implemented yet. Temporary success.')
        time.sleep(1.0)
        return True

    def apriltag_align_delivery(self) -> bool:
        self.publish_state(MissionState.ALIGNING_DELIVERY)
        self.get_logger().warn('apriltag_align_delivery() not implemented yet. Temporary success.')
        time.sleep(1.0)
        return True

    # --------------------------------------------------
    # UR arm
    # --------------------------------------------------
    def request_pickup_from_ur(self) -> bool:
        self.publish_state(MissionState.WAITING_UR_LOAD)

        if not self.pickup_client.wait_for_service(timeout_sec=5.0):
            self.publish_state(MissionState.MISSION_FAILED, 'PICKUP_SERVICE_UNAVAILABLE')
            return False

        req = Trigger.Request()
        future = self.pickup_client.call_async(req)
        result = self.wait_for_future(future, timeout_sec=180.0)

        if result is None:
            self.publish_state(MissionState.MISSION_FAILED, 'PICKUP_FAILED')
            return False

        if result.success:
            self.publish_state(MissionState.WAITING_UR_LOAD, 'ITEM_LOADED')
            return True

        self.publish_state(MissionState.MISSION_FAILED, f'PICKUP_FAILED:{result.message}')
        return False

    def request_unload_to_ur(self) -> bool:
        self.publish_state(MissionState.WAITING_UR_UNLOAD)

        if not self.unload_client.wait_for_service(timeout_sec=5.0):
            self.publish_state(MissionState.MISSION_FAILED, 'UNLOAD_SERVICE_UNAVAILABLE')
            return False

        req = Trigger.Request()
        future = self.unload_client.call_async(req)
        result = self.wait_for_future(future, timeout_sec=180.0)

        if result is None:
            self.publish_state(MissionState.MISSION_FAILED, 'UNLOAD_FAILED')
            return False

        if result.success:
            self.publish_state(MissionState.WAITING_UR_UNLOAD, 'ITEM_UNLOADED')
            return True

        self.publish_state(MissionState.MISSION_FAILED, f'UNLOAD_FAILED:{result.message}')
        return False

    # --------------------------------------------------
    # Waiting logic
    # --------------------------------------------------
    def wait_for_command_at_delivery(self) -> str:
        self.publish_state(MissionState.WAITING_AT_DELIVERY)
        self.waiting_for_delivery_command = True
        self.pending_order = None
        self.order_event.clear()

        start_time = time.time()
        while not self.cancel_requested:
            elapsed = time.time() - start_time

            if self.order_event.wait(timeout=0.2):
                break

            if elapsed >= self.delivery_wait_time:
                self.pending_order = 'timeout_return_to_charger'
                break

        self.waiting_for_delivery_command = False

        if self.cancel_requested:
            self.publish_state(MissionState.MISSION_CANCELLED)
            return 'cancelled'

        return self.pending_order or 'timeout_return_to_charger'

    def wait_with_status(self, state: MissionState, seconds: float) -> bool:
        self.publish_state(state)
        start_time = time.time()
        while time.time() - start_time < seconds:
            if self.cancel_requested:
                self.publish_state(MissionState.MISSION_CANCELLED)
                return False
            time.sleep(0.1)
        return True

    # --------------------------------------------------
    # Workflow helpers
    # --------------------------------------------------
    def go_to_pickup_station(self) -> bool:
        self.publish_state(MissionState.GOING_TO_PICKUP)

        if 'point_b_nav' in self.points:
            if not self.navigate_to_named_point('point_b_nav'):
                return False
        elif 'point_b' in self.points:
            if not self.navigate_to_named_point('point_b'):
                return False
        else:
            self.publish_state(MissionState.UNKNOWN_POINT, 'point_b_nav')
            return False

        if 'point_b_dock' in self.points:
            if not self.navigate_to_named_point('point_b_dock'):
                return False

        return self.apriltag_align_pickup()

    def go_to_delivery_station(self) -> bool:
        self.publish_state(MissionState.GOING_TO_DELIVERY)

        if 'point_a_nav' in self.points:
            if not self.navigate_to_named_point('point_a_nav'):
                return False
        elif 'point_a' in self.points:
            if not self.navigate_to_named_point('point_a'):
                return False
        else:
            self.publish_state(MissionState.UNKNOWN_POINT, 'point_a_nav')
            return False

        return self.apriltag_align_delivery()

    def go_to_charger_and_dock(self) -> bool:
        self.publish_state(MissionState.RETURNING_TO_CHARGER)

        if 'charger_approach' not in self.points:
            self.publish_state(MissionState.MISSION_FAILED, 'NO_CHARGER_APPROACH')
            return False

        if not self.navigate_to_named_point('charger_approach'):
            return False

        return self.auto_dock_to_charger()

    # --------------------------------------------------
    # Missions
    # --------------------------------------------------
    def run_prepare_only(self):
        with self.mission_lock:
            self.mission_running = True
            self.cancel_requested = False
            try:
                self.publish_state(MissionState.MISSION_STARTED)

                if not self.set_initial_pose_auto():
                    return

                self.publish_state(MissionState.READY_FOR_DELIVERY)
                time.sleep(1.0)

            except Exception as e:
                self.get_logger().error(f'Prepare-only mission error: {e}')
                self.publish_state(MissionState.MISSION_FAILED, str(e))

            finally:
                self.finish_mission(publish_idle=False)

    def run_go_to_named_point(self, point_name: str):
        with self.mission_lock:
            self.mission_running = True
            self.cancel_requested = False
            try:
                self.publish_state(MissionState.MISSION_STARTED)

                if not self.navigate_to_named_point(point_name):
                    return

                self.publish_state(MissionState.MISSION_COMPLETE)
                time.sleep(1.0)

            except Exception as e:
                self.get_logger().error(f'Unexpected error in go_to mission: {e}')
                self.publish_state(MissionState.MISSION_FAILED, str(e))

            finally:
                self.finish_mission()

    def run_dynamic_goal_mission(self):
        with self.mission_lock:
            self.mission_running = True
            self.cancel_requested = False
            try:
                self.publish_state(MissionState.MISSION_STARTED)

                if self.pending_dynamic_goal is None:
                    self.publish_state(MissionState.MISSION_FAILED, 'NO_DYNAMIC_GOAL')
                    return

                if not self.navigate_to_pose_stamped(self.pending_dynamic_goal, 'dynamic_goal'):
                    return

                self.publish_state(MissionState.MISSION_COMPLETE)
                time.sleep(1.0)

            except Exception as e:
                self.get_logger().error(f'Unexpected error in dynamic goal mission: {e}')
                self.publish_state(MissionState.MISSION_FAILED, str(e))

            finally:
                self.finish_mission()

    def run_task_list_mission(self):
        with self.mission_lock:
            self.mission_running = True
            self.cancel_requested = False
            try:
                self.publish_state(MissionState.MISSION_STARTED)

                if not self.pending_task_list:
                    self.publish_state(MissionState.MISSION_FAILED, 'NO_TASK_LIST')
                    return

                for point_name in self.pending_task_list:
                    if self.cancel_requested:
                        self.publish_state(MissionState.MISSION_CANCELLED)
                        return

                    if not self.navigate_to_named_point(point_name):
                        return

                self.publish_state(MissionState.MISSION_COMPLETE)
                time.sleep(1.0)

            except Exception as e:
                self.get_logger().error(f'Unexpected error in task list mission: {e}')
                self.publish_state(MissionState.MISSION_FAILED, str(e))

            finally:
                self.finish_mission()

    def run_dock_only(self):
        with self.mission_lock:
            self.mission_running = True
            self.cancel_requested = False
            try:
                self.publish_state(MissionState.MISSION_STARTED)

                if not self.go_to_charger_and_dock():
                    return

                self.publish_state(MissionState.MISSION_COMPLETE)
                time.sleep(1.0)

            except Exception as e:
                self.get_logger().error(f'Dock-only mission error: {e}')
                self.publish_state(MissionState.MISSION_FAILED, str(e))

            finally:
                self.finish_mission()

    def run_go_charge_only(self):
        with self.mission_lock:
            self.mission_running = True
            self.cancel_requested = False
            try:
                self.publish_state(MissionState.MISSION_STARTED)

                if not self.go_to_charger_and_dock():
                    return

                self.publish_state(MissionState.MISSION_COMPLETE)
                time.sleep(1.0)

            except Exception as e:
                self.get_logger().error(f'Go-charge mission error: {e}')
                self.publish_state(MissionState.MISSION_FAILED, str(e))

            finally:
                self.finish_mission()

    def run_default_delivery(self):
        with self.mission_lock:
            self.mission_running = True
            self.cancel_requested = False
            self.pending_order = None
            self.order_event.clear()

            try:
                self.publish_state(MissionState.MISSION_STARTED)

                if not self.go_to_pickup_station():
                    return

                if not self.request_pickup_from_ur():
                    return

                if not self.go_to_delivery_station():
                    return

                if not self.wait_with_status(MissionState.DELIVERING_ITEM, self.short_delivery_display_time):
                    return

                while not self.cancel_requested:
                    order = self.wait_for_command_at_delivery()

                    if order == 'cancelled':
                        return

                    if order in ('timeout_return_to_charger', 'go_charge'):
                        if not self.go_to_charger_and_dock():
                            return
                        break

                    if order == 'start_delivery':
                        if not self.go_to_pickup_station():
                            return

                        if not self.request_pickup_from_ur():
                            return

                        if not self.go_to_delivery_station():
                            return

                        if not self.wait_with_status(MissionState.DELIVERING_ITEM, self.short_delivery_display_time):
                            return

                        continue

                    if order == 'return_items':
                        if not self.go_to_pickup_station():
                            return

                        if not self.request_unload_to_ur():
                            return

                        if not self.go_to_delivery_station():
                            return

                        if not self.wait_with_status(MissionState.DELIVERING_ITEM, self.short_delivery_display_time):
                            return

                        continue

                self.publish_state(MissionState.MISSION_COMPLETE)
                time.sleep(1.0)

            except Exception as e:
                self.get_logger().error(f'Unexpected error in default mission: {e}')
                self.publish_state(MissionState.MISSION_FAILED, str(e))

            finally:
                self.finish_mission()

    def finish_mission(self, publish_idle: bool = True):
        self.waiting_for_delivery_command = False
        self.order_event.set()
        self.current_goal_handle = None
        self.mission_running = False
        self.cancel_requested = False
        self.stop_robot()
        if publish_idle:
            self.publish_state(MissionState.IDLE)


def main(args=None):
    rclpy.init(args=args)
    node = DeliveryMissionController()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down mission controller.')
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
