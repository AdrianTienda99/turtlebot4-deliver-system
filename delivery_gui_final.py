#!/usr/bin/env python3

import os
import sys
import math
import time
import threading
import yaml
import json

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
    QMessageBox, QTextEdit, QComboBox, QFrame, QSizePolicy, QLineEdit,
    QListWidget, QListWidgetItem, QScrollArea
)
from PyQt5.QtCore import QTimer, Qt, QRectF, QPointF
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
        self.feedback_sub = self.create_subscription(String, '/delivery_feedback', self.feedback_callback, 10)
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.pose_callback, 10)
        self.plan_sub = self.create_subscription(Path, '/plan', self.plan_callback, 10)
        self.battery_sub = self.create_subscription(BatteryState, '/battery_state', self.battery_callback, 10)

        self.current_status = 'UNKNOWN'
        self.current_status_base = 'UNKNOWN'
        self.last_command = 'NONE'

        self.feedback = {
            'state': 'UNKNOWN',
            'current_command': 'NONE',
            'mission_step': 'No feedback yet',
            'mission_running': False,
            'cancel_requested': False,
            'pending_order': None,
            'waiting_for_delivery_command': False,
            'waiting_for_ur_load': False,
            'waiting_for_ur_unload': False,
            'waiting_at_delivery': False,
            'returning_to_charge': False,
            'auto_docking': False,
            'tag_target_id': None,
            'tag_current_id': None,
            'tag_state': None,
            'tag_visible': False,
            'tag_aligned': False,
            'tag_error_x': None,
            'tag_width': None,
            'last_align_status_time': 0.0
        }

        self.robot_x = None
        self.robot_y = None
        self.robot_yaw = None
        self.last_pose_time = None

        self.plan_points = []

        self.battery_percentage = None
        self.battery_voltage = None

        self.status_log = []
        self.max_log_lines = 800

    def add_log(self, text: str):
        timestamp = time.strftime('%H:%M:%S')
        self.status_log.append(f'[{timestamp}] {text}')
        self.status_log = self.status_log[-self.max_log_lines:]

    def status_callback(self, msg: String):
        self.current_status = msg.data
        self.current_status_base = msg.data.split(':', 1)[0].strip()
        self.add_log(f'STATUS -> {msg.data}')

    def feedback_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
            self.feedback.update(data)

            feedback_state = str(self.feedback.get('state', 'UNKNOWN'))
            if feedback_state:
                self.current_status_base = feedback_state

            current_command = self.feedback.get('current_command')
            if current_command:
                self.last_command = str(current_command)

        except Exception as e:
            self.add_log(f'FEEDBACK PARSE ERROR -> {e}')

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

        self.map_margin = 8
        self.viewport_padding = 360

        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(900, 620)
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
            return 'DELIVERY STATION'
        if 'point_b' in lname:
            return 'PICK UP STATION'
        if 'charger' in lname:
            return 'CHARGER'
        return name.upper()

    def get_point_style(self, name: str):
        lname = name.lower()
        if 'point_a' in lname:
            return QColor(220, 38, 38)
        elif 'point_b' in lname:
            return QColor(37, 99, 235)
        elif 'charger' in lname:
            return QColor(249, 115, 22)
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

    def _collect_interest_pixels(self):
        pts = []

        for name, point in self.points.items():
            if self.should_draw_point(name) or 'point_b' in name.lower():
                pts.append(self.world_to_map_pixel(point['x'], point['y']))

        if self.ros_node.robot_x is not None and self.ros_node.robot_y is not None:
            pts.append(self.world_to_map_pixel(self.ros_node.robot_x, self.ros_node.robot_y))

        if self.clicked_goal_world is not None:
            pts.append(self.world_to_map_pixel(*self.clicked_goal_world))

        for p in self.ros_node.plan_points:
            pts.append(self.world_to_map_pixel(*p))

        return pts

    def get_viewport_rect(self):
        pts = self._collect_interest_pixels()
        if not pts:
            return QRectF(0, 0, self.map_width, self.map_height)

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        min_x = max(0.0, min(xs) - self.viewport_padding)
        max_x = min(float(self.map_width), max(xs) + self.viewport_padding)
        min_y = max(0.0, min(ys) - self.viewport_padding)
        max_y = min(float(self.map_height), max(ys) + self.viewport_padding)

        rect_w = max(50.0, max_x - min_x)
        rect_h = max(50.0, max_y - min_y)

        widget_ratio = max(1.0, self.width() - 2 * self.map_margin) / max(1.0, self.height() - 2 * self.map_margin)
        rect_ratio = rect_w / rect_h

        if rect_ratio > widget_ratio:
            desired_h = rect_w / widget_ratio
            grow = max(0.0, (desired_h - rect_h) / 2.0)
            min_y -= grow
            max_y += grow
        else:
            desired_w = rect_h * widget_ratio
            grow = max(0.0, (desired_w - rect_w) / 2.0)
            min_x -= grow
            max_x += grow

        if min_x < 0:
            max_x -= min_x
            min_x = 0
        if min_y < 0:
            max_y -= min_y
            min_y = 0
        if max_x > self.map_width:
            shift = max_x - self.map_width
            min_x -= shift
            max_x = self.map_width
        if max_y > self.map_height:
            shift = max_y - self.map_height
            min_y -= shift
            max_y = self.map_height

        min_x = max(0.0, min_x)
        min_y = max(0.0, min_y)

        return QRectF(min_x, min_y, max_x - min_x, max_y - min_y)

    def get_target_rect(self):
        return QRectF(
            self.map_margin,
            self.map_margin,
            max(1, self.width() - 2 * self.map_margin),
            max(1, self.height() - 2 * self.map_margin)
        )

    def world_to_view_pixel(self, x_world, y_world):
        x_pix, y_pix = self.world_to_map_pixel(x_world, y_world)
        viewport = self.get_viewport_rect()
        target = self.get_target_rect()

        rx = (x_pix - viewport.left()) / viewport.width()
        ry = (y_pix - viewport.top()) / viewport.height()

        vx = target.left() + rx * target.width()
        vy = target.top() + ry * target.height()
        return vx, vy

    def event_to_world(self, event):
        target = self.get_target_rect()
        viewport = self.get_viewport_rect()

        px = event.x()
        py = event.y()

        if not target.contains(QPointF(px, py)):
            return None

        rx = (px - target.left()) / target.width()
        ry = (py - target.top()) / target.height()

        map_x = viewport.left() + rx * viewport.width()
        map_y = viewport.top() + ry * viewport.height()

        return self.map_pixel_to_world(map_x, map_y)

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

    def clamp_text_position(self, x, y, text, painter, target_rect):
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(text)
        th = fm.height()

        x = max(target_rect.left() + 4, min(x, target_rect.right() - tw - 4))
        y = max(target_rect.top() + th, min(y, target_rect.bottom() - 4))
        return x, y

    def draw_named_points(self, painter):
        font = QFont('Arial', 13, QFont.Bold)
        painter.setFont(font)
        target = self.get_target_rect()

        for name, point in self.points.items():
            if not self.should_draw_point(name):
                continue

            x_pix, y_pix = self.world_to_view_pixel(point['x'], point['y'])
            color = self.get_point_style(name)
            selected = (name == self.selected_point)

            outer_r = 10 if selected else 8
            inner_r = 4 if selected else 3

            if selected:
                painter.setPen(QPen(QColor(16, 185, 129), 1.5))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(QRectF(x_pix - 11, y_pix - 11, 22, 22))

            painter.setPen(QPen(color, 1.4))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(QRectF(x_pix - outer_r, y_pix - outer_r, outer_r * 2, outer_r * 2))

            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.drawEllipse(QRectF(x_pix - inner_r, y_pix - inner_r, inner_r * 2, inner_r * 2))

            label = self.get_display_name(name)
            tx = x_pix + 16
            ty = y_pix - 8
            tx, ty = self.clamp_text_position(tx, ty, label, painter, target)
            painter.setPen(QPen(Qt.black, 1.0))
            painter.drawText(int(tx), int(ty), label)

    def draw_plan(self, painter):
        if len(self.ros_node.plan_points) < 2:
            return

        path = QPainterPath()
        first_x, first_y = self.world_to_view_pixel(*self.ros_node.plan_points[0])
        path.moveTo(first_x, first_y)

        for xw, yw in self.ros_node.plan_points[1:]:
            xp, yp = self.world_to_view_pixel(xw, yw)
            path.lineTo(xp, yp)

        painter.setPen(QPen(QColor(59, 130, 246, 190), 2.2))
        painter.drawPath(path)

    def draw_robot(self, painter):
        if self.ros_node.robot_x is None or self.ros_node.robot_y is None:
            return

        x_pix, y_pix = self.world_to_view_pixel(self.ros_node.robot_x, self.ros_node.robot_y)

        painter.setPen(QPen(QColor(37, 99, 235), 1.4))
        painter.setBrush(QBrush(QColor(37, 99, 235)))
        painter.drawEllipse(QRectF(x_pix - 6, y_pix - 6, 12, 12))

        if self.ros_node.robot_yaw is not None:
            arrow_len = 20
            end_x = x_pix + arrow_len * math.cos(self.ros_node.robot_yaw)
            end_y = y_pix - arrow_len * math.sin(self.ros_node.robot_yaw)
            painter.setPen(QPen(QColor(220, 38, 38), 2.0))
            painter.drawLine(int(x_pix), int(y_pix), int(end_x), int(end_y))

    def draw_clicked_goal(self, painter):
        if self.clicked_goal_world is None:
            return

        x_world, y_world = self.clicked_goal_world
        x_pix, y_pix = self.world_to_view_pixel(x_world, y_world)

        painter.setPen(QPen(QColor(124, 58, 237), 1.8))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QRectF(x_pix - 7, y_pix - 7, 14, 14))

        arrow_len = 18
        end_x = x_pix + arrow_len * math.cos(self.clicked_goal_yaw)
        end_y = y_pix - arrow_len * math.sin(self.clicked_goal_yaw)
        painter.drawLine(int(x_pix), int(y_pix), int(end_x), int(end_y))

    def draw_arm_station(self, painter):
        b_point = self.get_b_reference_point()
        if b_point is None:
            return

        bx, by = self.world_to_view_pixel(b_point['x'], b_point['y'])

        base_x = bx - 50
        base_y = by - 20

        painter.setPen(QPen(QColor(71, 85, 105), 1.2))
        painter.setBrush(QBrush(QColor(148, 163, 184)))
        painter.drawRect(QRectF(base_x, base_y, 24, 16))

        painter.setPen(QPen(QColor(30, 41, 59), 2.0))
        painter.drawLine(int(base_x + 9), int(base_y), int(base_x + 9), int(base_y - 18))
        painter.drawLine(int(base_x + 9), int(base_y - 18), int(base_x + 21), int(base_y - 28))
        painter.drawLine(int(base_x + 21), int(base_y - 28), int(base_x + 15), int(base_y - 40))

        font = QFont('Arial', 7, QFont.Bold)
        painter.setFont(font)
        painter.setPen(QPen(QColor(30, 41, 59), 1.0))
        painter.drawText(int(base_x - 2), int(base_y + 24), 'UR')

    def paintEvent(self, event):
        super().paintEvent(event)

        if self.map_pixmap is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        target = self.get_target_rect()
        viewport = self.get_viewport_rect()

        painter.drawPixmap(target, self.map_pixmap, viewport)

        self.draw_plan(painter)
        self.draw_named_points(painter)
        self.draw_arm_station(painter)
        self.draw_robot(painter)
        self.draw_clicked_goal(painter)


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

    def base_button_style(self):
        return """
            QPushButton {
                background-color: #ffffff;
                border: 1px solid #d0d7e2;
                border-radius: 10px;
                padding: 6px 10px;
                font-size: 12px;
                font-weight: bold;
                min-height: 34px;
                max-height: 34px;
            }
            QPushButton:hover { background-color: #f7f9fc; }
            QPushButton:pressed { background-color: #e9edf5; }
            QPushButton:disabled {
                background-color: #f3f4f6;
                color: #9ca3af;
                border: 1px solid #e5e7eb;
            }
        """

    def colored_button_style(self, bg, border):
        return f"""
            QPushButton {{
                background-color: {bg};
                border: 1px solid {border};
                border-radius: 10px;
                padding: 6px 10px;
                font-size: 12px;
                font-weight: bold;
                min-height: 34px;
                max-height: 34px;
            }}
            QPushButton:hover {{ background-color: #f8fafc; }}
            QPushButton:pressed {{ background-color: #eef2f7; }}
            QPushButton:disabled {{
                background-color: #f3f4f6;
                color: #9ca3af;
                border: 1px solid #e5e7eb;
            }}
        """

    def section_title_style(self):
        return 'font-size: 13px; font-weight: bold; color: #111827; background: transparent; padding: 0px;'

    def info_label_style(self):
        return 'font-size: 12px; color: #374151; background: transparent;'

    def panel_style(self):
        return (
            'QFrame { background: #ffffff; border-radius: 18px; border: 1px solid #dbe2ea; }'
        )

    def compact_card_style(self):
        return (
            'QFrame { background: #f8fafc; border-radius: 14px; border: 1px solid #dbe2ea; }'
        )

    def init_ui(self):
        self.setWindowTitle('TurtleBot 4 Delivery System')
        self.setGeometry(10, 10, 1880, 1020)
        self.setMinimumSize(1380, 820)
        self.setStyleSheet("""
            QWidget { background-color: #eef2f7; font-family: Arial; color: #1f2937; }
            QComboBox, QLineEdit {
                background-color: #ffffff;
                border: 1px solid #d0d7e2;
                border-radius: 9px;
                padding: 6px 8px;
                font-size: 12px;
                min-height: 32px;
                max-height: 32px;
            }
            QTextEdit {
                background-color: #071634;
                color: #e2e8f0;
                border-radius: 16px;
                padding: 10px;
                font-family: monospace;
                font-size: 12px;
            }
            QListWidget {
                background-color: #ffffff;
                border: 1px solid #d0d7e2;
                border-radius: 12px;
                padding: 4px;
                font-size: 12px;
            }
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background: #f1f5f9;
                width: 10px;
                margin: 0px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #cbd5e1;
                min-height: 24px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        title = QLabel('TurtleBot 4 Delivery System')
        title.setStyleSheet('font-size: 29px; font-weight: bold; background: transparent; color: #1f2937;')

        subtitle = QLabel('Map, Mission Feedback, Battery, Tasking, Request Form')
        subtitle.setStyleSheet('font-size: 13px; color: #6b7280; background: transparent;')

        self.status_indicator = QLabel('UNKNOWN')
        self.status_indicator.setAlignment(Qt.AlignCenter)
        self.status_indicator.setFixedHeight(40)
        self.status_indicator.setStyleSheet(self.status_style('UNKNOWN'))

        self.health_indicator = QLabel('HEALTH: UNKNOWN')
        self.health_indicator.setAlignment(Qt.AlignCenter)
        self.health_indicator.setFixedHeight(34)
        self.health_indicator.setStyleSheet(self.health_style('UNKNOWN'))

        self.battery_label = QLabel('Battery: --')
        self.last_command_label = QLabel('Last Command: NONE')
        self.pose_label = QLabel('Robot Pose: x=--, y=--, yaw=--')
        self.clicked_goal_label = QLabel('Clicked Goal: none')

        for label in [self.battery_label, self.last_command_label, self.pose_label, self.clicked_goal_label]:
            label.setStyleSheet(self.info_label_style())
            label.setWordWrap(True)

        self.feedback_state_label = QLabel('Feedback State: --')
        self.feedback_command_label = QLabel('Current Command: --')
        self.feedback_step_label = QLabel('Mission Step: --')
        self.feedback_tag_label = QLabel('Tag: --')
        self.feedback_flags_label = QLabel('Flags: --')

        for label in [
            self.feedback_state_label,
            self.feedback_command_label,
            self.feedback_step_label,
            self.feedback_tag_label,
            self.feedback_flags_label
        ]:
            label.setStyleSheet(self.info_label_style())
            label.setWordWrap(True)

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

        self.undock_button = QPushButton('Undock / Prepare Robot')
        self.undock_button.clicked.connect(lambda: self.send_gui_command('undock_robot'))
        self.undock_button.setStyleSheet(self.colored_button_style('#e0f2fe', '#7dd3fc'))

        self.dock_button = QPushButton('Dock Robot')
        self.dock_button.clicked.connect(lambda: self.send_gui_command('dock_robot'))
        self.dock_button.setStyleSheet(self.colored_button_style('#dcfce7', '#86efac'))

        self.start_button = QPushButton('Start Delivery')
        self.start_button.clicked.connect(lambda: self.send_gui_command('start_delivery'))
        self.start_button.setStyleSheet(self.base_button_style())

        self.return_items_button = QPushButton('Return Items')
        self.return_items_button.clicked.connect(lambda: self.send_gui_command('return_items'))
        self.return_items_button.setStyleSheet(self.colored_button_style('#fef3c7', '#fcd34d'))

        self.go_charge_button = QPushButton('Go Charge')
        self.go_charge_button.clicked.connect(lambda: self.send_gui_command('go_charge'))
        self.go_charge_button.setStyleSheet(self.colored_button_style('#dcfce7', '#86efac'))

        self.go_selected_button = QPushButton('Go to Selected Point')
        self.go_selected_button.clicked.connect(self.go_to_selected_point)
        self.go_selected_button.setStyleSheet(self.base_button_style())

        self.send_clicked_goal_button = QPushButton('Send Clicked Goal')
        self.send_clicked_goal_button.clicked.connect(self.send_clicked_goal)
        self.send_clicked_goal_button.setStyleSheet(self.colored_button_style('#ede9fe', '#c4b5fd'))

        self.clear_clicked_goal_button = QPushButton('Clear Clicked Goal')
        self.clear_clicked_goal_button.clicked.connect(self.clear_clicked_goal)
        self.clear_clicked_goal_button.setStyleSheet(self.base_button_style())

        self.stop_button = QPushButton('Stop Mission')
        self.stop_button.clicked.connect(lambda: self.send_gui_command('stop'))
        self.stop_button.setStyleSheet(self.colored_button_style('#fee2e2', '#fecaca'))

        self.save_name_input = QLineEdit()
        self.save_name_input.setPlaceholderText('new_station_name')

        self.save_clicked_point_button = QPushButton('Save Clicked Point to YAML')
        self.save_clicked_point_button.clicked.connect(self.save_clicked_point)
        self.save_clicked_point_button.setStyleSheet(self.base_button_style())

        self.reload_points_button = QPushButton('Reload Points')
        self.reload_points_button.clicked.connect(self.reload_points)
        self.reload_points_button.setStyleSheet(self.base_button_style())

        self.task_list_widget = QListWidget()

        self.add_selected_to_task_button = QPushButton('Add Selected Point')
        self.add_selected_to_task_button.clicked.connect(self.add_selected_to_task_list)
        self.add_selected_to_task_button.setStyleSheet(self.base_button_style())

        self.remove_task_button = QPushButton('Remove Selected Task')
        self.remove_task_button.clicked.connect(self.remove_selected_task)
        self.remove_task_button.setStyleSheet(self.base_button_style())

        self.clear_task_list_button = QPushButton('Clear Task List')
        self.clear_task_list_button.clicked.connect(self.task_list_widget.clear)
        self.clear_task_list_button.setStyleSheet(self.base_button_style())

        self.run_task_list_button = QPushButton('Run Task List Mission')
        self.run_task_list_button.clicked.connect(self.run_task_list_mission)
        self.run_task_list_button.setStyleSheet(self.colored_button_style('#dbeafe', '#93c5fd'))

        self.save_template_button = QPushButton('Save Current Task List')
        self.save_template_button.clicked.connect(self.save_template)
        self.save_template_button.setStyleSheet(self.base_button_style())

        self.load_template_button = QPushButton('Load Template to Task List')
        self.load_template_button.clicked.connect(self.load_template_to_task_list)
        self.load_template_button.setStyleSheet(self.base_button_style())

        self.run_template_button = QPushButton('Run Selected Template')
        self.run_template_button.clicked.connect(self.run_selected_template)
        self.run_template_button.setStyleSheet(self.colored_button_style('#dcfce7', '#86efac'))

        self.create_request_button = QPushButton('Create Delivery Request Mission')
        self.create_request_button.clicked.connect(self.create_delivery_request)
        self.create_request_button.setStyleSheet(self.base_button_style())

        self.run_request_button = QPushButton('Run Delivery Request Now')
        self.run_request_button.clicked.connect(self.run_delivery_request)
        self.run_request_button.setStyleSheet(self.colored_button_style('#fef3c7', '#fcd34d'))

        # LEFT SIDE
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(8)

        status_frame = QFrame()
        status_frame.setStyleSheet(self.compact_card_style())
        status_layout = QVBoxLayout()
        status_layout.setContentsMargins(10, 10, 10, 10)
        status_layout.setSpacing(6)
        status_title = QLabel('Mission Status')
        status_title.setStyleSheet(self.section_title_style())
        status_layout.addWidget(status_title)
        status_layout.addWidget(self.status_indicator)
        status_layout.addWidget(self.health_indicator)
        status_layout.addWidget(self.battery_label)
        status_layout.addWidget(self.last_command_label)
        status_layout.addWidget(self.pose_label)
        status_layout.addWidget(self.clicked_goal_label)
        status_frame.setLayout(status_layout)

        feedback_frame = QFrame()
        feedback_frame.setStyleSheet(self.compact_card_style())
        feedback_layout = QVBoxLayout()
        feedback_layout.setContentsMargins(10, 10, 10, 10)
        feedback_layout.setSpacing(5)
        feedback_title = QLabel('Live Mission Feedback')
        feedback_title.setStyleSheet(self.section_title_style())
        feedback_layout.addWidget(feedback_title)
        feedback_layout.addWidget(self.feedback_state_label)
        feedback_layout.addWidget(self.feedback_command_label)
        feedback_layout.addWidget(self.feedback_step_label)
        feedback_layout.addWidget(self.feedback_tag_label)
        feedback_layout.addWidget(self.feedback_flags_label)
        feedback_frame.setLayout(feedback_layout)

        points_frame = QFrame()
        points_frame.setStyleSheet(self.compact_card_style())
        points_layout = QVBoxLayout()
        points_layout.setContentsMargins(10, 10, 10, 10)
        points_layout.setSpacing(6)
        points_title = QLabel('Saved Delivery Points')
        points_title.setStyleSheet(self.section_title_style())
        points_layout.addWidget(points_title)
        points_layout.addWidget(self.point_selector)
        points_layout.addWidget(self.go_selected_button)
        points_layout.addWidget(self.send_clicked_goal_button)
        points_layout.addWidget(self.clear_clicked_goal_button)
        points_frame.setLayout(points_layout)

        save_point_frame = QFrame()
        save_point_frame.setStyleSheet(self.compact_card_style())
        save_point_layout = QVBoxLayout()
        save_point_layout.setContentsMargins(10, 10, 10, 10)
        save_point_layout.setSpacing(6)
        save_point_title = QLabel('Save New Point')
        save_point_title.setStyleSheet(self.section_title_style())
        save_point_layout.addWidget(save_point_title)
        save_point_layout.addWidget(self.save_name_input)
        save_point_layout.addWidget(self.save_clicked_point_button)
        save_point_layout.addWidget(self.reload_points_button)
        save_point_frame.setLayout(save_point_layout)

        quick_actions_frame = QFrame()
        quick_actions_frame.setStyleSheet(self.compact_card_style())
        quick_actions_layout = QVBoxLayout()
        quick_actions_layout.setContentsMargins(10, 10, 10, 10)
        quick_actions_layout.setSpacing(6)
        quick_actions_title = QLabel('Robot Quick Actions')
        quick_actions_title.setStyleSheet(self.section_title_style())
        quick_actions_layout.addWidget(quick_actions_title)
        quick_actions_layout.addWidget(self.undock_button)
        quick_actions_layout.addWidget(self.dock_button)
        quick_actions_layout.addWidget(self.start_button)
        quick_actions_layout.addWidget(self.return_items_button)
        quick_actions_layout.addWidget(self.go_charge_button)
        quick_actions_layout.addWidget(self.stop_button)
        quick_actions_frame.setLayout(quick_actions_layout)

        left_layout.addWidget(status_frame)
        left_layout.addWidget(feedback_frame)
        left_layout.addWidget(points_frame)
        left_layout.addWidget(save_point_frame)
        left_layout.addWidget(quick_actions_frame)
        left_layout.addStretch()

        left_content = QWidget()
        left_content.setLayout(left_layout)
        left_content.setStyleSheet('background: transparent;')

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setWidget(left_content)
        left_scroll.setFixedWidth(310)

        # CENTER
        self.map_view = MapView(self.map_yaml_path, self.points_config_path, self.ros_node)

        center_layout = QVBoxLayout()
        center_layout.setSpacing(8)
        center_layout.addWidget(title)
        center_layout.addWidget(subtitle)
        center_layout.addWidget(self.map_view, stretch=5)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(200)
        self.log_box.setMaximumHeight(280)

        log_title = QLabel('Mission Log')
        log_title.setStyleSheet(self.section_title_style())
        center_layout.addWidget(log_title)
        center_layout.addWidget(self.log_box, stretch=1)

        # RIGHT SIDE
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(8)

        request_frame = QFrame()
        request_frame.setStyleSheet(self.panel_style())
        request_layout = QVBoxLayout()
        request_layout.setContentsMargins(12, 12, 12, 12)
        request_layout.setSpacing(6)
        request_title = QLabel('Delivery Request Form')
        request_title.setStyleSheet(self.section_title_style())
        request_layout.addWidget(request_title)

        for text in ['Item Name', 'Pickup Point', 'Destination Point']:
            lbl = QLabel(text)
            lbl.setStyleSheet(self.info_label_style())
            request_layout.addWidget(lbl)
            if text == 'Item Name':
                request_layout.addWidget(self.request_item_input)
            elif text == 'Pickup Point':
                request_layout.addWidget(self.pickup_selector)
            else:
                request_layout.addWidget(self.destination_selector)

        request_layout.addWidget(self.create_request_button)
        request_layout.addWidget(self.run_request_button)
        request_frame.setLayout(request_layout)

        task_frame = QFrame()
        task_frame.setStyleSheet(self.panel_style())
        task_layout = QVBoxLayout()
        task_layout.setContentsMargins(12, 12, 12, 12)
        task_layout.setSpacing(6)
        task_title = QLabel('Task List Mission')
        task_title.setStyleSheet(self.section_title_style())
        task_layout.addWidget(task_title)
        task_layout.addWidget(self.task_list_widget)
        task_layout.addWidget(self.add_selected_to_task_button)
        task_layout.addWidget(self.remove_task_button)
        task_layout.addWidget(self.clear_task_list_button)
        task_layout.addWidget(self.run_task_list_button)
        task_frame.setLayout(task_layout)

        template_frame = QFrame()
        template_frame.setStyleSheet(self.panel_style())
        template_layout = QVBoxLayout()
        template_layout.setContentsMargins(12, 12, 12, 12)
        template_layout.setSpacing(6)
        template_title = QLabel('Mission Templates')
        template_title.setStyleSheet(self.section_title_style())
        template_layout.addWidget(template_title)
        template_layout.addWidget(self.template_selector)
        template_layout.addWidget(self.template_name_input)
        template_layout.addWidget(self.save_template_button)
        template_layout.addWidget(self.load_template_button)
        template_layout.addWidget(self.run_template_button)
        template_frame.setLayout(template_layout)

        right_layout.addWidget(request_frame)
        right_layout.addWidget(task_frame, stretch=1)
        right_layout.addWidget(template_frame)
        right_layout.addStretch()

        right_content = QWidget()
        right_content.setLayout(right_layout)
        right_content.setStyleSheet('background: transparent;')

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right_scroll.setWidget(right_content)
        right_scroll.setFixedWidth(330)

        main_layout = QHBoxLayout()
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.addWidget(left_scroll)
        main_layout.addLayout(center_layout, stretch=1)
        main_layout.addWidget(right_scroll)

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
        status_base = self.ros_node.current_status_base

        if status_base in ['MISSION_FAILED', 'STOP_REQUESTED', 'MISSION_CANCELLED']:
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
            'border-radius: 12px;'
            'font-size: 13px;'
            'font-weight: bold;'
            'padding: 6px;'
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
            'WAITING_AT_DELIVERY': '#fef3c7',
            'WAITING_UR_LOAD': '#fef3c7',
            'WAITING_UR_UNLOAD': '#fef3c7',
            'GOING_TO_PICKUP': '#dbeafe',
            'GOING_TO_DELIVERY': '#dbeafe',
            'RETURNING_TO_CHARGER': '#dcfce7',
            'AUTO_DOCKING': '#dcfce7',
            'ALIGNING_PICKUP': '#ede9fe',
            'ALIGNING_DELIVERY': '#ede9fe',
            'READY_FOR_DELIVERY': '#bbf7d0',
            'SETTING_INITIAL_POSE': '#dbeafe',
            'UNKNOWN': '#f3f4f6'
        }
        bg = colors.get(status, '#fef3c7')
        return (
            f'background-color: {bg};'
            'border: 1px solid #cbd5e1;'
            'border-radius: 12px;'
            'font-size: 15px;'
            'font-weight: bold;'
            'padding: 7px;'
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

    def build_feedback_flags_text(self):
        fb = self.ros_node.feedback
        flags = []

        if fb.get('mission_running'):
            flags.append('mission_running')
        if fb.get('cancel_requested'):
            flags.append('cancel_requested')
        if fb.get('waiting_for_ur_load'):
            flags.append('waiting_ur_load')
        if fb.get('waiting_for_ur_unload'):
            flags.append('waiting_ur_unload')
        if fb.get('waiting_at_delivery'):
            flags.append('waiting_at_delivery')
        if fb.get('returning_to_charge'):
            flags.append('returning_to_charge')
        if fb.get('auto_docking'):
            flags.append('auto_docking')
        if fb.get('waiting_for_delivery_command'):
            flags.append('waiting_for_command')

        return ', '.join(flags) if flags else 'none'

    def build_tag_feedback_text(self):
        fb = self.ros_node.feedback

        target_id = fb.get('tag_target_id')
        current_id = fb.get('tag_current_id')
        tag_state = fb.get('tag_state')
        visible = fb.get('tag_visible')
        aligned = fb.get('tag_aligned')
        error_x = fb.get('tag_error_x')
        tag_width = fb.get('tag_width')

        parts = [
            f'target={target_id if target_id is not None else "--"}',
            f'current={current_id if current_id is not None else "--"}',
            f'state={tag_state if tag_state else "--"}',
            f'visible={"yes" if visible else "no"}',
            f'aligned={"yes" if aligned else "no"}'
        ]

        if error_x is not None:
            parts.append(f'error_x={float(error_x):.1f}')
        if tag_width is not None:
            parts.append(f'width={float(tag_width):.1f}')

        return ' | '.join(parts)

    def update_button_states(self):
        status = self.ros_node.current_status_base
        mission_running = bool(self.ros_node.feedback.get('mission_running', False))
        clicked_goal_exists = self.map_view.clicked_goal_world is not None
        task_count = self.task_list_widget.count()

        busy_states = {
            'INITIALIZING',
            'MISSION_STARTED',
            'GOING_TO_PICKUP',
            'GOING_TO_DELIVERY',
            'ALIGNING_PICKUP',
            'ALIGNING_DELIVERY',
            'RETURNING_TO_CHARGER',
            'AUTO_DOCKING',
            'WAITING_UR_LOAD',
            'WAITING_UR_UNLOAD',
            'WAITING_AT_DELIVERY',
            'BUSY'
        }

        is_busy = mission_running or (status in busy_states)

        self.start_button.setEnabled(not is_busy)
        self.return_items_button.setEnabled(not is_busy)
        self.go_selected_button.setEnabled(not is_busy)
        self.run_task_list_button.setEnabled((task_count > 0) and (not is_busy))
        self.run_template_button.setEnabled((self.template_selector.count() > 0) and (not is_busy))
        self.run_request_button.setEnabled(not is_busy)
        self.create_request_button.setEnabled(True)

        self.send_clicked_goal_button.setEnabled(clicked_goal_exists and not is_busy)
        self.clear_clicked_goal_button.setEnabled(clicked_goal_exists)
        self.save_clicked_point_button.setEnabled(clicked_goal_exists and not is_busy)

        self.stop_button.setEnabled(is_busy or status in ['WAITING_AT_DELIVERY', 'WAITING_UR_LOAD', 'WAITING_UR_UNLOAD'])
        self.undock_button.setEnabled(not is_busy)
        self.dock_button.setEnabled(not is_busy)
        self.go_charge_button.setEnabled(not is_busy)

    def update_ui(self):
        status_raw = self.ros_node.current_status
        status_base = self.ros_node.current_status_base

        self.status_indicator.setText(status_raw)
        self.status_indicator.setStyleSheet(self.status_style(status_base))

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

        fb = self.ros_node.feedback
        self.feedback_state_label.setText(f'Feedback State: {fb.get("state", "--")}')
        self.feedback_command_label.setText(f'Current Command: {fb.get("current_command", "--")}')
        self.feedback_step_label.setText(f'Mission Step: {fb.get("mission_step", "--")}')
        self.feedback_tag_label.setText(f'Tag: {self.build_tag_feedback_text()}')
        self.feedback_flags_label.setText(f'Flags: {self.build_feedback_flags_text()}')

        self.update_button_states()
        self.map_view.update()

        log_text = '\n'.join(self.ros_node.status_log[-220:])
        if self.log_box.toPlainText() != log_text:
            scroll_bar = self.log_box.verticalScrollBar()
            at_bottom = scroll_bar.value() >= scroll_bar.maximum() - 4

            self.log_box.setPlainText(log_text)

            if at_bottom:
                scroll_bar.setValue(scroll_bar.maximum())

        if status_base == 'MISSION_COMPLETE' and self.last_popup_status != 'MISSION_COMPLETE':
            QMessageBox.information(self, 'Mission Complete', 'The delivery mission finished successfully.')
        elif status_base == 'MISSION_FAILED' and self.last_popup_status != 'MISSION_FAILED':
            QMessageBox.warning(self, 'Mission Failed', f'The delivery mission failed.\n\n{status_raw}')
        elif status_base == 'MISSION_CANCELLED' and self.last_popup_status != 'MISSION_CANCELLED':
            QMessageBox.information(self, 'Mission Cancelled', 'The mission was cancelled.')

        self.last_popup_status = status_base


def ros_spin(node):
    rclpy.spin(node)


def main():
    rclpy.init()

    map_yaml_path = '/root/final_map_tb4.yaml'
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
