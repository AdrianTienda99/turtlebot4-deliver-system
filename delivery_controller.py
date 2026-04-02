#!/usr/bin/env python3

import time
import threading
import yaml

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from std_msgs.msg import String
from std_srvs.srv import Trigger
from geometry_msgs.msg import Twist, PoseStamped
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus


class DeliveryMissionController(Node):
    def __init__(self):
        super().__init__('delivery_mission_controller')

        self.callback_group = ReentrantCallbackGroup()

        self.status_pub = self.create_publisher(String, '/delivery_status', 10)

        self.command_sub = self.create_subscription(
            String,
            '/delivery_command',
            self.command_callback,
            10,
            callback_group=self.callback_group
        )

        self.task_list_sub = self.create_subscription(
            String,
            '/delivery_task_list',
            self.task_list_callback,
            10,
            callback_group=self.callback_group
        )

        self.dynamic_goal_sub = self.create_subscription(
            PoseStamped,
            '/delivery_dynamic_goal',
            self.dynamic_goal_callback,
            10,
            callback_group=self.callback_group
        )

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.reload_srv = self.create_service(
            Trigger,
            '/reload_delivery_points',
            self.reload_points_callback,
            callback_group=self.callback_group
        )

        self.nav_client = ActionClient(
            self,
            NavigateToPose,
            'navigate_to_pose',
            callback_group=self.callback_group
        )

        self.pickup_client = self.create_client(
            Trigger,
            '/pickup_item',
            callback_group=self.callback_group
        )

        self.config_path = '/root/delivery_robot_project/config/delivery_points.yaml'
        self.points = {}
        self.load_points()

        self.delivery_wait_time = 2.0

        self.current_goal_handle = None
        self.mission_running = False
        self.cancel_requested = False
        self.mission_lock = threading.Lock()

        self.is_docked_at_pickup = False
        self.is_docked_at_delivery = False

        self.waiting_for_object_command = False
        self.pending_order = None
        self.order_event = threading.Event()

        self.pending_task_list = []
        self.pending_dynamic_goal = None

        self.publish_status('INITIALIZING')
        self.get_logger().info('Waiting for Nav2 action server...')
        while not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn('Still waiting for Nav2 action server...')
        self.get_logger().info('Nav2 action server available.')
        self.publish_status('IDLE')

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

    def publish_status(self, status_text: str):
        msg = String()
        msg.data = status_text
        self.status_pub.publish(msg)
        self.get_logger().info(f'[STATUS] {status_text}')

    def task_list_callback(self, msg: String):
        raw = msg.data.strip()
        if not raw:
            self.pending_task_list = []
            self.publish_status('TASK_LIST_EMPTY')
            return

        self.pending_task_list = [x.strip() for x in raw.split(',') if x.strip()]
        self.get_logger().info(f'Received task list: {self.pending_task_list}')
        self.publish_status('TASK_LIST_RECEIVED')

    def dynamic_goal_callback(self, msg: PoseStamped):
        self.pending_dynamic_goal = msg
        self.get_logger().info(
            f'Received dynamic goal: x={msg.pose.position.x:.3f}, y={msg.pose.position.y:.3f}'
        )
        self.publish_status('DYNAMIC_GOAL_RECEIVED')

    def command_callback(self, msg: String):
        command = msg.data.strip().lower()
        self.get_logger().info(f'Received command: {command}')

        if command == 'stop':
            self.stop_current_mission()
            return

        if self.waiting_for_object_command:
            if command == 'small_box':
                self.pending_order = 'small_box'
                self.waiting_for_object_command = False
                self.order_event.set()
                self.publish_status('ORDER_RECEIVED_SMALL_BOX')
            else:
                self.publish_status('WAITING_FOR_OBJECT_COMMAND')
            return

        if self.mission_running:
            self.publish_status('BUSY')
            return

        if command == 'start_delivery':
            threading.Thread(target=self.run_default_delivery, daemon=True).start()

        elif command == 'dock_at_b':
            threading.Thread(target=self.run_pickup_docking_only, daemon=True).start()

        elif command == 'dock_at_a':
            threading.Thread(target=self.run_delivery_docking_only, daemon=True).start()

        elif command.startswith('go_to:'):
            point_name = command.split(':', 1)[1].strip()
            threading.Thread(target=self.run_go_to_named_point, args=(point_name,), daemon=True).start()

        elif command == 'run_task_list':
            threading.Thread(target=self.run_task_list_mission, daemon=True).start()

        elif command == 'run_dynamic_goal':
            threading.Thread(target=self.run_dynamic_goal_mission, daemon=True).start()

        else:
            self.publish_status('UNKNOWN_COMMAND')

    def stop_current_mission(self):
        self.cancel_requested = True
        self.waiting_for_object_command = False
        self.order_event.set()
        self.publish_status('STOP_REQUESTED')

        if self.current_goal_handle is not None:
            self.current_goal_handle.cancel_goal_async()

        self.stop_robot()

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

    def navigate_to_pose_stamped(self, pose: PoseStamped, label='dynamic_goal') -> bool:
        if self.cancel_requested:
            self.publish_status('MISSION_CANCELLED')
            return False

        self.publish_status(f'GOING_TO_{label.upper()}')

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        send_goal_future = self.nav_client.send_goal_async(goal_msg)
        goal_handle = self.wait_for_future(send_goal_future, timeout_sec=20.0)

        if goal_handle is None:
            self.publish_status('MISSION_CANCELLED' if self.cancel_requested else 'MISSION_FAILED')
            return False

        self.current_goal_handle = goal_handle

        if not goal_handle.accepted:
            self.current_goal_handle = None
            self.publish_status('MISSION_FAILED')
            return False

        result_future = goal_handle.get_result_async()
        result = self.wait_for_future(result_future, timeout_sec=600.0)
        self.current_goal_handle = None

        if result is None:
            self.publish_status('MISSION_CANCELLED' if self.cancel_requested else 'MISSION_FAILED')
            return False

        if result.status == GoalStatus.STATUS_SUCCEEDED:
            self.stop_robot()
            time.sleep(1.0)
            return True

        if result.status == GoalStatus.STATUS_CANCELED:
            self.publish_status('MISSION_CANCELLED')
            return False

        self.publish_status('MISSION_FAILED')
        return False

    def navigate_to_named_point(self, point_name: str, point: dict) -> bool:
        if self.cancel_requested:
            self.publish_status('MISSION_CANCELLED')
            return False

        if point_name not in ['point_b_nav', 'point_b_dock']:
            self.is_docked_at_pickup = False

        if point_name not in ['point_a_nav', 'point_a']:
            self.is_docked_at_delivery = False

        self.publish_status(f'GOING_TO_{point_name.upper()}')

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = self.pose_from_point(point)

        self.get_logger().info(
            f"Sending goal to {point_name}: x={point['x']:.3f}, y={point['y']:.3f}"
        )

        send_goal_future = self.nav_client.send_goal_async(goal_msg)
        goal_handle = self.wait_for_future(send_goal_future, timeout_sec=20.0)

        if goal_handle is None:
            self.publish_status('MISSION_CANCELLED' if self.cancel_requested else 'MISSION_FAILED')
            return False

        self.current_goal_handle = goal_handle

        if not goal_handle.accepted:
            self.publish_status('MISSION_FAILED')
            self.current_goal_handle = None
            return False

        result_future = goal_handle.get_result_async()
        result = self.wait_for_future(result_future, timeout_sec=600.0)
        self.current_goal_handle = None

        if result is None:
            self.publish_status('MISSION_CANCELLED' if self.cancel_requested else 'MISSION_FAILED')
            return False

        status = result.status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.stop_robot()
            time.sleep(1.0)
            self.get_logger().info(f'Successfully reached {point_name}.')
            return True

        if status == GoalStatus.STATUS_CANCELED:
            self.publish_status('MISSION_CANCELLED')
            return False

        self.publish_status('MISSION_FAILED')
        return False

    def stop_robot(self):
        msg = Twist()
        self.cmd_vel_pub.publish(msg)

    def drive_for_time(self, linear_x=0.0, angular_z=0.0, duration=1.0):
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z

        start_time = time.time()
        rate_hz = 20.0
        dt = 1.0 / rate_hz

        while (time.time() - start_time) < duration:
            if self.cancel_requested:
                break
            self.cmd_vel_pub.publish(msg)
            time.sleep(dt)

        self.stop_robot()
        time.sleep(0.6)

    def final_docking_b(self):
        if self.cancel_requested:
            self.publish_status('MISSION_CANCELLED')
            return False

        if self.is_docked_at_pickup:
            self.publish_status('ALREADY_DOCKED_AT_B')
            return True

        self.publish_status('FINAL_DOCKING_B')
        self.stop_robot()
        time.sleep(1.0)

        angular_speed = 0.25
        linear_speed = 0.03

        turn_90_time = 1.40 / angular_speed
        forward_right_time = 0.20 / linear_speed
        forward_front_time = 0.40 / linear_speed

        tiny_left_angle = 0.30
        tiny_left_time = tiny_left_angle / angular_speed

        self.get_logger().info('Dock B step 1: rotate right slightly less')
        self.drive_for_time(0.0, -angular_speed, turn_90_time)
        if self.cancel_requested:
            return False

        self.get_logger().info('Dock B step 2: move right')
        self.drive_for_time(linear_speed, 0.0, forward_right_time)
        if self.cancel_requested:
            return False

        self.get_logger().info('Dock B step 3: rotate back left')
        self.drive_for_time(0.0, angular_speed, turn_90_time)
        if self.cancel_requested:
            return False

        self.get_logger().info('Dock B step 4: move forward')
        self.drive_for_time(linear_speed, 0.0, forward_front_time)
        if self.cancel_requested:
            return False

        self.get_logger().info('Dock B step 5: stronger left straighten')
        self.drive_for_time(0.0, angular_speed, tiny_left_time)
        if self.cancel_requested:
            return False

        self.is_docked_at_pickup = True
        self.publish_status('DOCKED_AT_PICKUP_STATION')
        return True

    def final_docking_a(self):
        if self.cancel_requested:
            self.publish_status('MISSION_CANCELLED')
            return False

        if self.is_docked_at_delivery:
            self.publish_status('ALREADY_DOCKED_AT_A')
            return True

        self.publish_status('FINAL_DOCKING_A')
        self.stop_robot()
        time.sleep(1.0)

        linear_speed = 0.03
        angular_speed = 0.20

        forward_time = 0.17 / linear_speed
        tiny_left_angle = 0.30
        tiny_left_time = tiny_left_angle / angular_speed

        self.get_logger().info('Dock A step 1: stronger left align')
        self.drive_for_time(0.0, angular_speed, tiny_left_time)
        if self.cancel_requested:
            return False

        self.get_logger().info('Dock A step 2: move forward')
        self.drive_for_time(linear_speed, 0.0, forward_time)
        if self.cancel_requested:
            return False

        self.is_docked_at_delivery = True
        self.publish_status('DOCKED_AT_DELIVERY_STATION')
        return True

    def go_to_b_nav_and_dock(self) -> bool:
        self.is_docked_at_pickup = False

        if 'point_b_nav' not in self.points:
            self.publish_status('UNKNOWN_POINT')
            return False

        if not self.navigate_to_named_point('point_b_nav', self.points['point_b_nav']):
            return False

        if 'point_b_dock' in self.points:
            if not self.navigate_to_named_point('point_b_dock', self.points['point_b_dock']):
                return False

        return self.final_docking_b()

    def go_to_a_nav_and_dock(self) -> bool:
        self.is_docked_at_delivery = False

        if 'point_a_nav' in self.points:
            if not self.navigate_to_named_point('point_a_nav', self.points['point_a_nav']):
                return False
        elif 'point_a' in self.points:
            if not self.navigate_to_named_point('point_a', self.points['point_a']):
                return False
        else:
            self.publish_status('UNKNOWN_POINT')
            return False

        return self.final_docking_a()

    def request_pickup_from_ur(self) -> bool:
        self.publish_status('WAITING_FOR_UR_ARM')

        if not self.pickup_client.wait_for_service(timeout_sec=5.0):
            self.publish_status('PICKUP_SERVICE_UNAVAILABLE')
            return False

        req = Trigger.Request()
        future = self.pickup_client.call_async(req)
        result = self.wait_for_future(future, timeout_sec=180.0)

        if result is None:
            self.publish_status('PICKUP_FAILED')
            return False

        if result.success:
            self.publish_status('ITEM_LOADED')
            return True

        self.publish_status('PICKUP_FAILED')
        return False

    def wait_for_order_at_a(self) -> bool:
        self.publish_status('WAITING_FOR_OBJECT_COMMAND')
        self.waiting_for_object_command = True
        self.pending_order = None
        self.order_event.clear()

        while not self.cancel_requested:
            if self.order_event.wait(timeout=0.2):
                break

        self.waiting_for_object_command = False

        if self.cancel_requested:
            self.publish_status('MISSION_CANCELLED')
            return False

        if self.pending_order == 'small_box':
            return True

        self.publish_status('MISSION_FAILED')
        return False

    def wait_with_status(self, status_text: str, seconds: float) -> bool:
        self.publish_status(status_text)
        start_time = time.time()
        while time.time() - start_time < seconds:
            if self.cancel_requested:
                self.publish_status('MISSION_CANCELLED')
                return False
            time.sleep(0.1)
        return True

    def run_go_to_named_point(self, point_name: str):
        with self.mission_lock:
            self.mission_running = True
            self.cancel_requested = False
            try:
                self.publish_status('MISSION_STARTED')

                if point_name not in self.points:
                    self.publish_status('UNKNOWN_POINT')
                    return

                if self.navigate_to_named_point(point_name, self.points[point_name]):
                    self.publish_status('MISSION_COMPLETE')
                    time.sleep(1.5)

            except Exception as e:
                self.get_logger().error(f'Unexpected error in go_to mission: {e}')
                self.publish_status('MISSION_FAILED')

            finally:
                self.current_goal_handle = None
                self.mission_running = False
                self.cancel_requested = False
                self.stop_robot()
                self.publish_status('IDLE')

    def run_dynamic_goal_mission(self):
        with self.mission_lock:
            self.mission_running = True
            self.cancel_requested = False
            try:
                self.publish_status('MISSION_STARTED')

                if self.pending_dynamic_goal is None:
                    self.publish_status('NO_DYNAMIC_GOAL')
                    return

                if self.navigate_to_pose_stamped(self.pending_dynamic_goal, 'dynamic_goal'):
                    self.publish_status('MISSION_COMPLETE')
                    time.sleep(1.5)

            except Exception as e:
                self.get_logger().error(f'Unexpected error in dynamic goal mission: {e}')
                self.publish_status('MISSION_FAILED')

            finally:
                self.current_goal_handle = None
                self.mission_running = False
                self.cancel_requested = False
                self.stop_robot()
                self.publish_status('IDLE')

    def run_task_list_mission(self):
        with self.mission_lock:
            self.mission_running = True
            self.cancel_requested = False
            try:
                self.publish_status('MISSION_STARTED')

                if not self.pending_task_list:
                    self.publish_status('NO_TASK_LIST')
                    return

                for point_name in self.pending_task_list:
                    if self.cancel_requested:
                        self.publish_status('MISSION_CANCELLED')
                        return

                    if point_name not in self.points:
                        self.publish_status('UNKNOWN_POINT')
                        return

                    if not self.navigate_to_named_point(point_name, self.points[point_name]):
                        return

                self.publish_status('MISSION_COMPLETE')
                time.sleep(1.5)

            except Exception as e:
                self.get_logger().error(f'Unexpected error in task list mission: {e}')
                self.publish_status('MISSION_FAILED')

            finally:
                self.current_goal_handle = None
                self.mission_running = False
                self.cancel_requested = False
                self.stop_robot()
                self.publish_status('IDLE')

    def run_pickup_docking_only(self):
        with self.mission_lock:
            self.mission_running = True
            self.cancel_requested = False
            self.is_docked_at_pickup = False
            try:
                self.publish_status('MISSION_STARTED')
                if self.go_to_b_nav_and_dock():
                    self.publish_status('MISSION_COMPLETE')
                    time.sleep(1.5)
            except Exception as e:
                self.get_logger().error(f'Unexpected error in dock B routine: {e}')
                self.publish_status('MISSION_FAILED')
            finally:
                self.current_goal_handle = None
                self.mission_running = False
                self.cancel_requested = False
                self.stop_robot()
                self.publish_status('IDLE')

    def run_delivery_docking_only(self):
        with self.mission_lock:
            self.mission_running = True
            self.cancel_requested = False
            self.is_docked_at_delivery = False
            try:
                self.publish_status('MISSION_STARTED')
                if self.go_to_a_nav_and_dock():
                    self.publish_status('MISSION_COMPLETE')
                    time.sleep(1.5)
            except Exception as e:
                self.get_logger().error(f'Unexpected error in dock A routine: {e}')
                self.publish_status('MISSION_FAILED')
            finally:
                self.current_goal_handle = None
                self.mission_running = False
                self.cancel_requested = False
                self.stop_robot()
                self.publish_status('IDLE')

    def run_default_delivery(self):
        with self.mission_lock:
            self.mission_running = True
            self.cancel_requested = False
            self.is_docked_at_pickup = False
            self.is_docked_at_delivery = False
            self.pending_order = None
            self.order_event.clear()

            try:
                self.publish_status('MISSION_STARTED')

                if not self.go_to_a_nav_and_dock():
                    return

                if not self.wait_for_order_at_a():
                    return

                if not self.go_to_b_nav_and_dock():
                    return

                if not self.request_pickup_from_ur():
                    return

                self.is_docked_at_pickup = False

                if not self.go_to_a_nav_and_dock():
                    return

                if not self.wait_with_status('DELIVERING_ITEM_AT_POINT_A', self.delivery_wait_time):
                    return

                self.stop_robot()
                self.publish_status('MISSION_COMPLETE')
                time.sleep(1.5)

            except Exception as e:
                self.get_logger().error(f'Unexpected error in mission: {e}')
                self.publish_status('MISSION_FAILED')

            finally:
                self.waiting_for_object_command = False
                self.order_event.set()
                self.current_goal_handle = None
                self.mission_running = False
                self.cancel_requested = False
                self.stop_robot()
                self.publish_status('IDLE')


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
