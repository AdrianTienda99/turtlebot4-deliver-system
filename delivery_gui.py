#!/usr/bin/env python3

import os
import sys
import math
import time
import threading
import yaml

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
    QMessageBox, QTextEdit, QComboBox, QFrame, QSizePolicy, QLineEdit,
    QListWidget, QListWidgetItem
)
from PyQt5.QtCore import QTimer, Qt, QRectF
from PyQt5.QtGui import (
    QPixmap, QPainter, QColor, QPen, QBrush, QFont, QPainterPath
)

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import BatteryState


class DeliveryGuiNode(Node):
    def __init__(self):
        super().__init__('delivery_gui_node')

        self.command_pub = self.create_publisher(String, '/delivery_command', 10)
        self.dynamic_goal_pub = self.create_publisher(PoseStamped, '/delivery_dynamic_goal', 10)
        self.task_list_pub = self.create_publisher(String, '/delivery_task_list', 10)

        self.reload_client = self.create_client(Trigger, '/reload_delivery_points')

        self.status_sub = self.create_subscription(String, '/delivery_status', self.status_callback, 10)
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.pose_callback, 10)
        self.plan_sub = self.create_subscription(Path, '/plan', self.plan_callback, 10)
        self.battery_sub = self.create_subscription(BatteryState, '/battery_state', self.battery_callback, 10)

        self.current_status = 'UNKNOWN'
        self.last_command = 'NONE'

        self.robot_x = None
        self.robot_y = None
        self.robot_yaw = None
        self.last_pose_time = None

        self.plan_points = []

        self.battery_percentage = None
        self.battery_voltage = None

        self.status_log = []
        self.max_log_lines = 600

    def add_log(self, text: str):
        timestamp = time.strftime('%H:%M:%S')
        self.status_log.append(f'[{timestamp}] {text}')
        self.status_log = self.status_log[-self.max_log_lines:]

    def status_callback(self, msg: String):
        self.current_status = msg.data
        self.add_log(f'STATUS -> {msg.data}')

    def pose_callback(self, msg: PoseWithCovarianceStamped):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.last_pose_time = time.time()

        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)

    def plan_callback(self, msg: Path):
        self.plan_points = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

    def battery_callback(self, msg: BatteryState):
        self.battery_voltage = msg.voltage

        if msg.percentage is not None and msg.percentage >= 0.0:
            if msg.percentage <= 1.0:
                self.battery_percentage = msg.percentage * 100.0
            else:
                self.battery_percentage = msg.percentage

    def send_command(self, command: str):
        msg = String()
        msg.data = command
        self.command_pub.publish(msg)
        self.last_command = command
        self.add_log(f'COMMAND -> {command}')

    def send_task_list(self, task_list: str):
        msg = String()
        msg.data = task_list
        self.task_list_pub.publish(msg)
        self.last_command = f'task_list: {task_list}'
        self.add_log(f'COMMAND -> task_list: {task_list}')

    def send_dynamic_goal(self, x: float, y: float, yaw: float = 0.0):
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = 0.0

        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)

        self.dynamic_goal_pub.publish(msg)
        self.last_command = f'dynamic_goal ({x:.2f}, {y:.2f}, yaw={math.degrees(yaw):.1f})'
        self.add_log(f'COMMAND -> dynamic_goal x={x:.2f}, y={y:.2f}, yaw={math.degrees(yaw):.1f}')

    def reload_points(self):
        if not self.reload_client.wait_for_service(timeout_sec=2.0):
            self.add_log('SERVICE -> reload_delivery_points unavailable')
            return False, 'Reload service unavailable.'

        req = Trigger.Request()
        future = self.reload_client.call_async(req)

        start = time.time()
        while not future.done() and (time.time() - start) < 5.0:
            time.sleep(0.1)

        if not future.done():
            return False, 'Reload service timed out.'

        result = future.result()
        return result.success, result.message


class MapView(QLabel):
    def __init__(self, map_yaml_path, points_config_path, ros_node: DeliveryGuiNode):
        super().__init__()
        self.ros_node = ros_node
        self.map_yaml_path = map_yaml_path
        self.points_config_path = points_config_path

        self.map_pixmap = None
        self.map_resolution = None
        self.map_origin = None
        self.map_image_path = None
        self.map_width = None
        self.map_height = None

        self.points = {}
        self.selected_point = None

        self.clicked_goal_world = None
        self.clicked_goal_yaw = 0.0
        self.dragging_orientation = False

        self.map_margin = 12

        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(1120, 820)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(
            'background-color: #ffffff; border-radius: 22px; border: 1px solid #d9e0ea;'
        )

        self.load_map()
        self.load_points()

    def load_map(self):
        with open(self.map_yaml_path, 'r') as f:
            map_data = yaml.safe_load(f)

        image_field = map_data['image']
        if os.path.isabs(image_field):
            self.map_image_path = image_field
        else:
            self.map_image_path = os.path.join(os.path.dirname(self.map_yaml_path), image_field)

        self.map_resolution = float(map_data['resolution'])
        self.map_origin = map_data['origin']

        self.map_pixmap = QPixmap(self.map_image_path)
        if self.map_pixmap.isNull():
            raise RuntimeError(f'Failed to load map image: {self.map_image_path}')

        self.map_width = self.map_pixmap.width()
        self.map_height = self.map_pixmap.height()

    def load_points(self):
        with open(self.points_config_path, 'r') as f:
            self.points = yaml.safe_load(f) or {}

    def set_selected_point(self, selected_name: str):
        self.selected_point = selected_name
        self.update()

    def get_display_name(self, name: str) -> str:
        lname = name.lower()
        if 'point_a' in lname:
            return 'A DEL'
        if 'point_b' in lname:
            return 'B PU'
        return name.upper()

    def get_point_style(self, name: str):
        lname = name.lower()
        if 'point_a' in lname:
            return QColor(220, 38, 38)
        elif 'point_b' in lname:
            return QColor(37, 99, 235)
        return QColor(249, 115, 22)

    def should_draw_point(self, name: str) -> bool:
        lname = name.lower()
        if lname == 'point_a_nav':
            return False
        if lname == 'point_b_nav':
            return False
        return True

    def get_b_reference_point(self):
        if 'point_b_dock' in self.points:
            return self.points['point_b_dock']
        if 'point_b_nav' in self.points:
            return self.points['point_b_nav']

        for name, point in self.points.items():
            if 'point_b' in name.lower():
                return point
        return None

    def get_visual_offset(self, name: str):
        """
        Visual-only shift so A and B markers appear inside the white area.
        This does NOT change the actual navigation coordinates.
        """
        lname = name.lower()
        if 'point_b' in lname:
            return (34.0, 0.0)   # shift B to the right
        if 'point_a' in lname:
            return (26.0, 0.0)   # shift A to the right
        return (0.0, 0.0)

    def world_to_map_pixel(self, x_world, y_world):
        origin_x = float(self.map_origin[0])
        origin_y = float(self.map_origin[1])

        x_pix = (x_world - origin_x) / self.map_resolution
        y_pix = (y_world - origin_y) / self.map_resolution
        y_pix = self.map_height - y_pix
        return x_pix, y_pix

    def map_pixel_to_world(self, x_pix, y_pix):
        origin_x = float(self.map_origin[0])
        origin_y = float(self.map_origin[1])

        y_pix_flipped = self.map_height - y_pix

        x_world = origin_x + x_pix * self.map_resolution
        y_world = origin_y + y_pix_flipped * self.map_resolution
        return x_world, y_world

    def get_scaled_geometry(self):
        available_w = max(1, self.width() - 2 * self.map_margin)
        available_h = max(1, self.height() - 2 * self.map_margin)

        scaled = self.map_pixmap.scaled(
            available_w,
            available_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        x_offset = (self.width() - scaled.width()) // 2
        y_offset = (self.height() - scaled.height()) // 2
        return scaled, x_offset, y_offset

    def event_to_world(self, event):
        scaled, x_offset, y_offset = self.get_scaled_geometry()

        px = event.x()
        py = event.y()

        if not (x_offset <= px <= x_offset + scaled.width() and y_offset <= py <= y_offset + scaled.height()):
            return None

        local_x = (px - x_offset) / (scaled.width() / self.map_width)
        local_y = (py - y_offset) / (scaled.height() / self.map_height)
        return self.map_pixel_to_world(local_x, local_y)

    def mousePressEvent(self, event):
        if self.map_pixmap is None:
            return

        world = self.event_to_world(event)
        if world is None:
            return

        x_world, y_world = world
        self.clicked_goal_world = (x_world, y_world)
        self.clicked_goal_yaw = 0.0
        self.dragging_orientation = True
        self.ros_node.add_log(f'MAP CLICK -> x={x_world:.2f}, y={y_world:.2f}')
        self.update()

    def mouseMoveEvent(self, event):
        if not self.dragging_orientation or self.clicked_goal_world is None:
            return

        world = self.event_to_world(event)
        if world is None:
            return

        gx, gy = self.clicked_goal_world
        mx, my = world
        self.clicked_goal_yaw = math.atan2(my - gy, mx - gx)
        self.update()

    def mouseReleaseEvent(self, event):
        self.dragging_orientation = False
        self.update()

    def clamp_text_position(self, x, y, text, painter):
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(text)
        th = fm.height()

        x = max(4, min(x, self.map_width - tw - 4))
        y = max(th, min(y, self.map_height - 4))
        return x, y

    def draw_named_points(self, painter):
        font = QFont('Arial', 6, QFont.Bold)
        painter.setFont(font)

        for name, point in self.points.items():
            if not self.should_draw_point(name):
                continue

            x_pix, y_pix = self.world_to_map_pixel(point['x'], point['y'])
            off_x, off_y = self.get_visual_offset(name)
            x_pix += off_x
            y_pix += off_y

            color = self.get_point_style(name)
            selected = (name == self.selected_point)

            outer_r = 4.5 if selected else 3.5
            inner_r = 1.8 if selected else 1.3

            if selected:
                painter.setPen(QPen(QColor(16, 185, 129), 1.0))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(QRectF(x_pix - 7, y_pix - 7, 14, 14))

            painter.setPen(QPen(color, 1.0))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(QRectF(x_pix - outer_r, y_pix - outer_r, outer_r * 2, outer_r * 2))

            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.drawEllipse(QRectF(x_pix - inner_r, y_pix - inner_r, inner_r * 2, inner_r * 2))

            label = self.get_display_name(name)
            tx = x_pix + 6
            ty = y_pix - 4
            tx, ty = self.clamp_text_position(tx, ty, label, painter)

            painter.setPen(QPen(Qt.black, 0.8))
            painter.drawText(int(tx), int(ty), label)

    def draw_plan(self, painter):
        if len(self.ros_node.plan_points) < 2:
            return

        path = QPainterPath()
        first_x, first_y = self.world_to_map_pixel(*self.ros_node.plan_points[0])
        path.moveTo(first_x, first_y)

        for xw, yw in self.ros_node.plan_points[1:]:
            xp, yp = self.world_to_map_pixel(xw, yw)
            path.lineTo(xp, yp)

        painter.setPen(QPen(QColor(59, 130, 246, 190), 1.3))
        painter.drawPath(path)

    def draw_robot(self, painter):
        if self.ros_node.robot_x is None or self.ros_node.robot_y is None:
            return

        x_pix, y_pix = self.world_to_map_pixel(self.ros_node.robot_x, self.ros_node.robot_y)

        painter.setPen(QPen(QColor(37, 99, 235), 1.2))
        painter.setBrush(QBrush(QColor(37, 99, 235)))
        painter.drawEllipse(QRectF(x_pix - 4, y_pix - 4, 8, 8))

        if self.ros_node.robot_yaw is not None:
            arrow_len = 12
            end_x = x_pix + arrow_len * math.cos(self.ros_node.robot_yaw)
            end_y = y_pix - arrow_len * math.sin(self.ros_node.robot_yaw)

            painter.setPen(QPen(QColor(220, 38, 38), 1.5))
            painter.drawLine(int(x_pix), int(y_pix), int(end_x), int(end_y))

    def draw_clicked_goal(self, painter):
        if self.clicked_goal_world is None:
            return

        x_world, y_world = self.clicked_goal_world
        x_pix, y_pix = self.world_to_map_pixel(x_world, y_world)

        painter.setPen(QPen(QColor(124, 58, 237), 1.5))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QRectF(x_pix - 6, y_pix - 6, 12, 12))

        arrow_len = 15
        end_x = x_pix + arrow_len * math.cos(self.clicked_goal_yaw)
        end_y = y_pix - arrow_len * math.sin(self.clicked_goal_yaw)
        painter.drawLine(int(x_pix), int(y_pix), int(end_x), int(end_y))

    def draw_arm_station(self, painter):
        b_point = self.get_b_reference_point()
        if b_point is None:
            return

        bx, by = self.world_to_map_pixel(b_point['x'], b_point['y'])

        # keep UR arm fixed as requested
        base_x = bx + 18
        base_y = by - 5

        painter.setPen(QPen(QColor(71, 85, 105), 1.0))
        painter.setBrush(QBrush(QColor(148, 163, 184)))
        painter.drawRect(QRectF(base_x, base_y, 11, 7))

        painter.setPen(QPen(QColor(30, 41, 59), 1.6))
        painter.drawLine(int(base_x + 5.5), int(base_y), int(base_x + 5.5), int(base_y - 9))
        painter.drawLine(int(base_x + 5.5), int(base_y - 9), int(base_x + 13), int(base_y - 14))
        painter.drawLine(int(base_x + 13), int(base_y - 14), int(base_x + 10), int(base_y - 19))

        font = QFont('Arial', 5, QFont.Bold)
        painter.setFont(font)
        painter.setPen(QPen(QColor(30, 41, 59), 0.8))
        tx, ty = self.clamp_text_position(base_x - 1, base_y + 13, 'UR', painter)
        painter.drawText(int(tx), int(ty), 'UR')

    def paintEvent(self, event):
        super().paintEvent(event)

        if self.map_pixmap is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        scaled, x_offset, y_offset = self.get_scaled_geometry()
        painter.drawPixmap(x_offset, y_offset, scaled)

        scale_x = scaled.width() / self.map_width
        scale_y = scaled.height() / self.map_height

        painter.save()
        painter.translate(x_offset, y_offset)
        painter.scale(scale_x, scale_y)

        self.draw_plan(painter)
        self.draw_named_points(painter)
        self.draw_arm_station(painter)
        self.draw_robot(painter)
        self.draw_clicked_goal(painter)

        painter.restore()


class DeliveryGui(QWidget):
    def __init__(self, ros_node: DeliveryGuiNode, map_yaml_path: str, points_config_path: str, templates_path: str):
        super().__init__()
        self.ros_node = ros_node
        self.map_yaml_path = map_yaml_path
        self.points_config_path = points_config_path
        self.templates_path = templates_path

        self.last_popup_status = None
        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(200)

    def init_ui(self):
        self.setWindowTitle('TurtleBot 4 Delivery System')
        self.setGeometry(10, 10, 1880, 1040)
        self.setStyleSheet("""
            QWidget { background-color: #eef2f7; font-family: Arial; color: #1f2937; }
            QPushButton {
                background-color: #ffffff; border: 1px solid #d0d7e2; border-radius: 12px;
                padding: 10px; font-size: 13px; font-weight: bold;
            }
            QPushButton:hover { background-color: #f7f9fc; }
            QPushButton:pressed { background-color: #e9edf5; }
            QComboBox, QLineEdit {
                background-color: #ffffff; border: 1px solid #d0d7e2; border-radius: 10px;
                padding: 8px; font-size: 13px;
            }
            QTextEdit {
                background-color: #071634; color: #e2e8f0; border-radius: 16px;
                padding: 10px; font-family: monospace; font-size: 12px;
            }
            QListWidget {
                background-color: #ffffff; border: 1px solid #d0d7e2; border-radius: 12px;
                padding: 6px; font-size: 13px;
            }
        """)

        title = QLabel('TurtleBot 4 Delivery System')
        title.setStyleSheet('font-size: 31px; font-weight: bold; background: transparent; color: #1f2937;')

        subtitle = QLabel('Advanced Mission Control, Health, Route Preview, and Pickup Visualization')
        subtitle.setStyleSheet('font-size: 14px; color: #6b7280; background: transparent;')

        self.status_indicator = QLabel('UNKNOWN')
        self.status_indicator.setAlignment(Qt.AlignCenter)
        self.status_indicator.setFixedHeight(42)
        self.status_indicator.setStyleSheet(self.status_style('UNKNOWN'))

        self.health_indicator = QLabel('HEALTH: UNKNOWN')
        self.health_indicator.setAlignment(Qt.AlignCenter)
        self.health_indicator.setFixedHeight(38)
        self.health_indicator.setStyleSheet(self.health_style('UNKNOWN'))

        self.battery_label = QLabel('Battery: --')
        self.last_command_label = QLabel('Last Command: NONE')
        self.pose_label = QLabel('Robot Pose: x=--, y=--, yaw=--')
        self.clicked_goal_label = QLabel('Clicked Goal: none')

        self.point_selector = QComboBox()
        self.template_selector = QComboBox()
        self.template_name_input = QLineEdit()
        self.template_name_input.setPlaceholderText('template_name')

        self.request_item_input = QLineEdit()
        self.request_item_input.setPlaceholderText('item name')
        self.pickup_selector = QComboBox()
        self.destination_selector = QComboBox()

        self.load_points_into_dropdown()
        self.load_templates_into_dropdown()

        self.point_selector.currentTextChanged.connect(self.on_point_changed)

        self.start_button = QPushButton('Start Delivery')
        self.start_button.clicked.connect(lambda: self.send_gui_command('start_delivery'))

        self.small_box_button = QPushButton('Small Box')
        self.small_box_button.clicked.connect(lambda: self.send_gui_command('small_box'))
        self.small_box_button.setStyleSheet(
            'background-color: #dbeafe; border: 1px solid #93c5fd; border-radius: 12px; padding: 10px; font-weight: bold;'
        )

        self.go_selected_button = QPushButton('Go to Selected Point')
        self.go_selected_button.clicked.connect(self.go_to_selected_point)

        self.send_clicked_goal_button = QPushButton('Send Clicked Goal')
        self.send_clicked_goal_button.clicked.connect(self.send_clicked_goal)
        self.send_clicked_goal_button.setStyleSheet(
            'background-color: #ede9fe; border: 1px solid #c4b5fd; border-radius: 12px; padding: 10px; font-weight: bold;'
        )

        self.clear_clicked_goal_button = QPushButton('Clear Clicked Goal')
        self.clear_clicked_goal_button.clicked.connect(self.clear_clicked_goal)

        self.stop_button = QPushButton('Stop Mission')
        self.stop_button.clicked.connect(lambda: self.send_gui_command('stop'))
        self.stop_button.setStyleSheet(
            'background-color: #fee2e2; border: 1px solid #fecaca; border-radius: 12px; padding: 10px; font-weight: bold;'
        )

        self.save_name_input = QLineEdit()
        self.save_name_input.setPlaceholderText('new_station_name')

        self.save_clicked_point_button = QPushButton('Save Clicked Point to YAML')
        self.save_clicked_point_button.clicked.connect(self.save_clicked_point)

        self.reload_points_button = QPushButton('Reload Points')
        self.reload_points_button.clicked.connect(self.reload_points)

        self.task_list_widget = QListWidget()

        self.add_selected_to_task_button = QPushButton('Add Selected Point to Task List')
        self.add_selected_to_task_button.clicked.connect(self.add_selected_to_task_list)

        self.remove_task_button = QPushButton('Remove Selected Task')
        self.remove_task_button.clicked.connect(self.remove_selected_task)

        self.clear_task_list_button = QPushButton('Clear Task List')
        self.clear_task_list_button.clicked.connect(self.task_list_widget.clear)

        self.run_task_list_button = QPushButton('Run Task List Mission')
        self.run_task_list_button.clicked.connect(self.run_task_list_mission)
        self.run_task_list_button.setStyleSheet(
            'background-color: #dbeafe; border: 1px solid #93c5fd; border-radius: 12px; padding: 10px; font-weight: bold;'
        )

        self.save_template_button = QPushButton('Save Current Task List as Template')
        self.save_template_button.clicked.connect(self.save_template)

        self.load_template_button = QPushButton('Load Template to Task List')
        self.load_template_button.clicked.connect(self.load_template_to_task_list)

        self.run_template_button = QPushButton('Run Selected Template')
        self.run_template_button.clicked.connect(self.run_selected_template)
        self.run_template_button.setStyleSheet(
            'background-color: #dcfce7; border: 1px solid #86efac; border-radius: 12px; padding: 10px; font-weight: bold;'
        )

        self.create_request_button = QPushButton('Create Delivery Request Mission')
        self.create_request_button.clicked.connect(self.create_delivery_request)

        self.run_request_button = QPushButton('Run Delivery Request Now')
        self.run_request_button.clicked.connect(self.run_delivery_request)
        self.run_request_button.setStyleSheet(
            'background-color: #fef3c7; border: 1px solid #fcd34d; border-radius: 12px; padding: 10px; font-weight: bold;'
        )

        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel('Mission Status'))
        left_layout.addWidget(self.status_indicator)
        left_layout.addWidget(self.health_indicator)
        left_layout.addSpacing(6)
        left_layout.addWidget(self.battery_label)
        left_layout.addWidget(self.last_command_label)
        left_layout.addWidget(self.pose_label)
        left_layout.addWidget(self.clicked_goal_label)
        left_layout.addSpacing(12)
        left_layout.addWidget(QLabel('Saved Delivery Points'))
        left_layout.addWidget(self.point_selector)
        left_layout.addWidget(self.go_selected_button)
        left_layout.addSpacing(10)
        left_layout.addWidget(self.send_clicked_goal_button)
        left_layout.addWidget(self.clear_clicked_goal_button)
        left_layout.addSpacing(10)
        left_layout.addWidget(QLabel('Save New Point'))
        left_layout.addWidget(self.save_name_input)
        left_layout.addWidget(self.save_clicked_point_button)
        left_layout.addWidget(self.reload_points_button)
        left_layout.addSpacing(10)
        left_layout.addWidget(self.start_button)
        left_layout.addWidget(self.small_box_button)
        left_layout.addWidget(self.stop_button)
        left_layout.addStretch()

        left_frame = QFrame()
        left_frame.setLayout(left_layout)
        left_frame.setFixedWidth(270)
        left_frame.setStyleSheet(
            'QFrame { background: #ffffff; border-radius: 22px; border: 1px solid #dbe2ea; padding: 12px; }'
        )

        self.map_view = MapView(self.map_yaml_path, self.points_config_path, self.ros_node)

        center_layout = QVBoxLayout()
        center_layout.addWidget(title)
        center_layout.addWidget(subtitle)
        center_layout.addSpacing(10)
        center_layout.addWidget(self.map_view, stretch=1)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(200)

        center_layout.addSpacing(8)
        center_layout.addWidget(QLabel('Mission Log'))
        center_layout.addWidget(self.log_box)

        right_layout = QVBoxLayout()

        request_frame = QFrame()
        request_frame.setStyleSheet(
            'QFrame { background: #ffffff; border-radius: 22px; border: 1px solid #dbe2ea; padding: 12px; }'
        )
        request_layout = QVBoxLayout()
        request_layout.addWidget(QLabel('Delivery Request Form'))
        request_layout.addWidget(QLabel('Item Name'))
        request_layout.addWidget(self.request_item_input)
        request_layout.addWidget(QLabel('Pickup Point'))
        request_layout.addWidget(self.pickup_selector)
        request_layout.addWidget(QLabel('Destination Point'))
        request_layout.addWidget(self.destination_selector)
        request_layout.addWidget(self.create_request_button)
        request_layout.addWidget(self.run_request_button)
        request_frame.setLayout(request_layout)

        task_frame = QFrame()
        task_frame.setStyleSheet(
            'QFrame { background: #ffffff; border-radius: 22px; border: 1px solid #dbe2ea; padding: 12px; }'
        )
        task_layout = QVBoxLayout()
        task_layout.addWidget(QLabel('Task List Mission'))
        task_layout.addWidget(self.task_list_widget)
        task_layout.addWidget(self.add_selected_to_task_button)
        task_layout.addWidget(self.remove_task_button)
        task_layout.addWidget(self.clear_task_list_button)
        task_layout.addWidget(self.run_task_list_button)
        task_frame.setLayout(task_layout)

        template_frame = QFrame()
        template_frame.setStyleSheet(
            'QFrame { background: #ffffff; border-radius: 22px; border: 1px solid #dbe2ea; padding: 12px; }'
        )
        template_layout = QVBoxLayout()
        template_layout.addWidget(QLabel('Mission Templates'))
        template_layout.addWidget(self.template_selector)
        template_layout.addWidget(self.template_name_input)
        template_layout.addWidget(self.save_template_button)
        template_layout.addWidget(self.load_template_button)
        template_layout.addWidget(self.run_template_button)
        template_frame.setLayout(template_layout)

        right_layout.addWidget(request_frame)
        right_layout.addWidget(task_frame, stretch=1)
        right_layout.addWidget(template_frame)

        right_frame = QFrame()
        right_frame.setLayout(right_layout)
        right_frame.setFixedWidth(315)
        right_frame.setStyleSheet('QFrame { background: transparent; border: none; }')

        main_layout = QHBoxLayout()
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.addWidget(left_frame)
        main_layout.addLayout(center_layout, stretch=1)
        main_layout.addWidget(right_frame)

        self.setLayout(main_layout)

    def load_points_from_yaml(self):
        with open(self.points_config_path, 'r') as f:
            return yaml.safe_load(f) or {}

    def load_templates_from_yaml(self):
        if not os.path.exists(self.templates_path):
            return {}
        with open(self.templates_path, 'r') as f:
            return yaml.safe_load(f) or {}

    def save_templates_to_yaml(self, templates):
        with open(self.templates_path, 'w') as f:
            yaml.safe_dump(templates, f, sort_keys=False)

    def load_points_into_dropdown(self):
        points = self.load_points_from_yaml()

        current_point = self.point_selector.currentText()
        current_pickup = self.pickup_selector.currentText()
        current_dest = self.destination_selector.currentText()

        self.point_selector.clear()
        self.pickup_selector.clear()
        self.destination_selector.clear()

        for name in points.keys():
            self.point_selector.addItem(name)
            self.pickup_selector.addItem(name)
            self.destination_selector.addItem(name)

        if current_point:
            idx = self.point_selector.findText(current_point)
            if idx >= 0:
                self.point_selector.setCurrentIndex(idx)

        if current_pickup:
            idx = self.pickup_selector.findText(current_pickup)
            if idx >= 0:
                self.pickup_selector.setCurrentIndex(idx)

        if current_dest:
            idx = self.destination_selector.findText(current_dest)
            if idx >= 0:
                self.destination_selector.setCurrentIndex(idx)

    def load_templates_into_dropdown(self):
        current_template = self.template_selector.currentText()
        self.template_selector.clear()
        templates = self.load_templates_from_yaml()
        for name in templates.keys():
            self.template_selector.addItem(name)

        if current_template:
            idx = self.template_selector.findText(current_template)
            if idx >= 0:
                self.template_selector.setCurrentIndex(idx)

    def derive_health_state(self):
        if self.ros_node.current_status in ['MISSION_FAILED', 'STOP_REQUESTED', 'MISSION_CANCELLED']:
            return 'ALERT'

        if self.ros_node.battery_percentage is not None and self.ros_node.battery_percentage < 20.0:
            return 'LOW_BATTERY'

        if self.ros_node.last_pose_time is None:
            return 'NO_LOCALIZATION'

        if (time.time() - self.ros_node.last_pose_time) > 3.0:
            return 'LOCALIZATION_LOST'

        return 'NORMAL'

    def health_style(self, health: str) -> str:
        colors = {
            'NORMAL': '#dcfce7',
            'LOW_BATTERY': '#fef3c7',
            'ALERT': '#fecaca',
            'NO_LOCALIZATION': '#e5e7eb',
            'LOCALIZATION_LOST': '#fecaca',
            'UNKNOWN': '#f3f4f6'
        }
        bg = colors.get(health, '#f3f4f6')
        return (
            f'background-color: {bg};'
            'border: 1px solid #cbd5e1;'
            'border-radius: 14px;'
            'font-size: 14px;'
            'font-weight: bold;'
            'padding: 8px;'
        )

    def status_style(self, status: str) -> str:
        colors = {
            'IDLE': '#e5e7eb',
            'INITIALIZING': '#dbeafe',
            'MISSION_STARTED': '#dbeafe',
            'MISSION_COMPLETE': '#bbf7d0',
            'MISSION_FAILED': '#fecaca',
            'MISSION_CANCELLED': '#fecaca',
            'STOP_REQUESTED': '#fecaca',
            'BUSY': '#fde68a',
            'UNKNOWN': '#f3f4f6'
        }
        bg = colors.get(status, '#fef3c7')
        return (
            f'background-color: {bg};'
            'border: 1px solid #cbd5e1;'
            'border-radius: 14px;'
            'font-size: 16px;'
            'font-weight: bold;'
            'padding: 8px;'
        )

    def on_point_changed(self, point_name: str):
        self.map_view.set_selected_point(point_name)

    def go_to_selected_point(self):
        selected = self.point_selector.currentText()
        if selected:
            self.send_gui_command(f'go_to:{selected}')

    def send_clicked_goal(self):
        if self.map_view.clicked_goal_world is None:
            QMessageBox.information(self, 'No Goal Selected', 'Click and drag on the map first.')
            return

        x_world, y_world = self.map_view.clicked_goal_world
        yaw = self.map_view.clicked_goal_yaw
        self.ros_node.send_dynamic_goal(x_world, y_world, yaw)
        self.ros_node.send_command('run_dynamic_goal')

    def clear_clicked_goal(self):
        self.map_view.clicked_goal_world = None
        self.map_view.clicked_goal_yaw = 0.0
        self.map_view.update()

    def save_clicked_point(self):
        if self.map_view.clicked_goal_world is None:
            QMessageBox.information(self, 'No Point Selected', 'Click on the map first.')
            return

        name = self.save_name_input.text().strip().lower()
        if not name:
            QMessageBox.information(self, 'Missing Name', 'Enter a point name first.')
            return

        x_world, y_world = self.map_view.clicked_goal_world
        yaw = self.map_view.clicked_goal_yaw

        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)

        try:
            points = self.load_points_from_yaml()
            points[name] = {
                'frame_id': 'map',
                'x': float(x_world),
                'y': float(y_world),
                'z': 0.0,
                'qx': 0.0,
                'qy': 0.0,
                'qz': float(qz),
                'qw': float(qw)
            }

            with open(self.points_config_path, 'w') as f:
                yaml.safe_dump(points, f, sort_keys=False)

            self.ros_node.add_log(f'SAVED POINT -> {name}')
            QMessageBox.information(self, 'Saved', f'Point "{name}" saved successfully.')
            self.reload_points()
            self.save_name_input.clear()
        except Exception as e:
            QMessageBox.critical(self, 'Save Error', str(e))

    def reload_points(self):
        ok, message = self.ros_node.reload_points()
        if not ok:
            QMessageBox.warning(self, 'Reload Failed', message)
            return

        self.map_view.load_points()
        self.load_points_into_dropdown()
        self.map_view.update()
        self.ros_node.add_log(f'POINTS RELOADED -> {message}')

    def add_selected_to_task_list(self):
        selected = self.point_selector.currentText()
        if selected:
            self.task_list_widget.addItem(QListWidgetItem(selected))

    def remove_selected_task(self):
        row = self.task_list_widget.currentRow()
        if row >= 0:
            self.task_list_widget.takeItem(row)

    def run_task_list_mission(self):
        items = [self.task_list_widget.item(i).text() for i in range(self.task_list_widget.count())]
        if not items:
            QMessageBox.information(self, 'Empty Task List', 'Add at least one point to the task list.')
            return

        task_string = ','.join(items)
        self.ros_node.send_task_list(task_string)
        self.ros_node.send_command('run_task_list')

    def save_template(self):
        template_name = self.template_name_input.text().strip()
        if not template_name:
            QMessageBox.information(self, 'Missing Name', 'Enter a template name.')
            return

        items = [self.task_list_widget.item(i).text() for i in range(self.task_list_widget.count())]
        if not items:
            QMessageBox.information(self, 'Empty Task List', 'Add points before saving a template.')
            return

        templates = self.load_templates_from_yaml()
        templates[template_name] = items
        self.save_templates_to_yaml(templates)
        self.load_templates_into_dropdown()
        self.ros_node.add_log(f'TEMPLATE SAVED -> {template_name}')
        QMessageBox.information(self, 'Template Saved', f'Template "{template_name}" saved successfully.')
        self.template_name_input.clear()

    def load_template_to_task_list(self):
        template_name = self.template_selector.currentText()
        if not template_name:
            QMessageBox.information(self, 'No Template', 'Select a template first.')
            return

        templates = self.load_templates_from_yaml()
        if template_name not in templates:
            QMessageBox.warning(self, 'Template Missing', 'Selected template was not found.')
            return

        self.task_list_widget.clear()
        for point_name in templates[template_name]:
            self.task_list_widget.addItem(QListWidgetItem(point_name))

        self.ros_node.add_log(f'TEMPLATE LOADED -> {template_name}')

    def run_selected_template(self):
        template_name = self.template_selector.currentText()
        if not template_name:
            QMessageBox.information(self, 'No Template', 'Select a template first.')
            return

        templates = self.load_templates_from_yaml()
        if template_name not in templates:
            QMessageBox.warning(self, 'Template Missing', 'Selected template was not found.')
            return

        task_string = ','.join(templates[template_name])
        self.ros_node.send_task_list(task_string)
        self.ros_node.send_command('run_task_list')

    def create_delivery_request(self):
        item = self.request_item_input.text().strip()
        pickup = self.pickup_selector.currentText()
        destination = self.destination_selector.currentText()

        if not item:
            QMessageBox.information(self, 'Missing Item', 'Enter an item name first.')
            return

        if not pickup or not destination:
            QMessageBox.information(self, 'Missing Points', 'Select pickup and destination points.')
            return

        self.task_list_widget.clear()
        self.task_list_widget.addItem(QListWidgetItem(pickup))
        self.task_list_widget.addItem(QListWidgetItem(destination))

        self.ros_node.add_log(f'REQUEST CREATED -> item={item}, pickup={pickup}, destination={destination}')
        QMessageBox.information(
            self,
            'Request Created',
            f'Item: {item}\nPickup: {pickup}\nDestination: {destination}\n\nTask list prepared.'
        )

    def run_delivery_request(self):
        item = self.request_item_input.text().strip()
        pickup = self.pickup_selector.currentText()
        destination = self.destination_selector.currentText()

        if not item:
            QMessageBox.information(self, 'Missing Item', 'Enter an item name first.')
            return

        if not pickup or not destination:
            QMessageBox.information(self, 'Missing Points', 'Select pickup and destination points.')
            return

        task_string = f'{pickup},{destination}'
        self.ros_node.add_log(f'REQUEST RUN -> item={item}, pickup={pickup}, destination={destination}')
        self.ros_node.send_task_list(task_string)
        self.ros_node.send_command('run_task_list')

    def send_gui_command(self, command: str):
        try:
            self.ros_node.send_command(command)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to send command:\n{e}')

    def update_ui(self):
        status = self.ros_node.current_status
        self.status_indicator.setText(status)
        self.status_indicator.setStyleSheet(self.status_style(status))

        health = self.derive_health_state()
        self.health_indicator.setText(f'HEALTH: {health}')
        self.health_indicator.setStyleSheet(self.health_style(health))

        if self.ros_node.battery_percentage is not None:
            if self.ros_node.battery_voltage is not None:
                self.battery_label.setText(
                    f'Battery: {self.ros_node.battery_percentage:.1f}% | {self.ros_node.battery_voltage:.2f} V'
                )
            else:
                self.battery_label.setText(f'Battery: {self.ros_node.battery_percentage:.1f}%')
        else:
            self.battery_label.setText('Battery: unavailable')

        self.last_command_label.setText(f'Last Command: {self.ros_node.last_command}')

        if self.ros_node.robot_x is not None and self.ros_node.robot_y is not None:
            yaw_deg = math.degrees(self.ros_node.robot_yaw) if self.ros_node.robot_yaw is not None else 0.0
            self.pose_label.setText(
                f'Robot Pose: x={self.ros_node.robot_x:.2f}, y={self.ros_node.robot_y:.2f}, yaw={yaw_deg:.1f}°'
            )
        else:
            self.pose_label.setText('Robot Pose: x=--, y=--, yaw=--')

        if self.map_view.clicked_goal_world is not None:
            xw, yw = self.map_view.clicked_goal_world
            yaw_deg = math.degrees(self.map_view.clicked_goal_yaw)
            self.clicked_goal_label.setText(f'Clicked Goal: x={xw:.2f}, y={yw:.2f}, yaw={yaw_deg:.1f}°')
        else:
            self.clicked_goal_label.setText('Clicked Goal: none')

        self.map_view.update()

        log_text = '\n'.join(self.ros_node.status_log[-180:])
        if self.log_box.toPlainText() != log_text:
            self.log_box.setPlainText(log_text)
            cursor = self.log_box.textCursor()
            cursor.movePosition(cursor.End)
            self.log_box.setTextCursor(cursor)

        if status == 'MISSION_COMPLETE' and self.last_popup_status != 'MISSION_COMPLETE':
            QMessageBox.information(self, 'Mission Complete', 'The delivery mission finished successfully.')
        elif status == 'MISSION_FAILED' and self.last_popup_status != 'MISSION_FAILED':
            QMessageBox.warning(self, 'Mission Failed', 'The delivery mission failed.')
        elif status == 'MISSION_CANCELLED' and self.last_popup_status != 'MISSION_CANCELLED':
            QMessageBox.information(self, 'Mission Cancelled', 'The mission was cancelled.')

        self.last_popup_status = status


def ros_spin(node):
    rclpy.spin(node)


def main():
    rclpy.init()

    map_yaml_path = '/my_map.yaml'
    points_config_path = '/root/delivery_robot_project/config/delivery_points.yaml'
    templates_path = '/root/delivery_robot_project/config/mission_templates.yaml'

    ros_node = DeliveryGuiNode()
    ros_thread = threading.Thread(target=ros_spin, args=(ros_node,), daemon=True)
    ros_thread.start()

    app = QApplication(sys.argv)
    gui = DeliveryGui(ros_node, map_yaml_path, points_config_path, templates_path)
    gui.show()

    exit_code = app.exec_()

    ros_node.destroy_node()
    rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
