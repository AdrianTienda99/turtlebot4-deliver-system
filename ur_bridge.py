#!/usr/bin/env python3

import socket
import time

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


UR_IP = "192.168.50.3"
UR_PORT = 29999
UR_PROGRAM = "adrian_delivery.urp"
UR_START_TIMEOUT = 8.0
UR_FINISH_TIMEOUT = 120.0
POLL_PERIOD = 0.5


class URBridge(Node):
    def __init__(self):
        super().__init__('ur_bridge')
        self.srv = self.create_service(Trigger, '/pickup_item', self.handle_pickup)
        self.get_logger().info('UR Bridge ready on /pickup_item')

    def send_dashboard_command(self, command, wait_time=0.3):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3.0)
        s.connect((UR_IP, UR_PORT))
        time.sleep(0.15)

        try:
            _ = s.recv(1024).decode(errors='ignore')
        except Exception:
            pass

        s.sendall((command + '\n').encode())
        time.sleep(wait_time)

        response = ""
        try:
            response = s.recv(4096).decode(errors='ignore').strip()
        except Exception:
            pass

        s.close()
        return response

    def get_program_state(self):
        return self.send_dashboard_command('programState', wait_time=0.2)

    def is_running(self):
        resp = self.send_dashboard_command('running', wait_time=0.2)
        return 'true' in resp.lower()

    def stop_program(self):
        return self.send_dashboard_command('stop', wait_time=0.4)

    def handle_pickup(self, request, response):
        self.get_logger().info('Pickup request -> loading and starting UR program')

        try:
            load_resp = self.send_dashboard_command(f'load {UR_PROGRAM}', wait_time=1.0)
            self.get_logger().info(f'Load response: {load_resp}')

            if 'failed' in load_resp.lower() or 'error' in load_resp.lower():
                response.success = False
                response.message = f'Failed to load program: {load_resp}'
                return response

            play_resp = self.send_dashboard_command('play', wait_time=0.8)
            self.get_logger().info(f'Play response: {play_resp}')

            if 'failed' in play_resp.lower() or 'not allowed' in play_resp.lower():
                response.success = False
                response.message = f'UR play failed: {play_resp}'
                return response

            # Wait for the program to actually start
            start_time = time.time()
            started = False
            while (time.time() - start_time) < UR_START_TIMEOUT:
                if self.is_running():
                    started = True
                    self.get_logger().info('UR program is running.')
                    break
                time.sleep(POLL_PERIOD)

            if not started:
                response.success = False
                response.message = 'UR program did not start running.'
                return response

            # Wait until the program finishes
            finish_start = time.time()
            while (time.time() - finish_start) < UR_FINISH_TIMEOUT:
                if not self.is_running():
                    state = self.get_program_state()
                    self.get_logger().info(f'UR finished. Final state: {state}')
                    response.success = True
                    response.message = 'UR pickup completed and robot returned home.'
                    return response
                time.sleep(POLL_PERIOD)

            # Timeout safety stop
            stop_resp = self.stop_program()
            self.get_logger().warn(f'UR finish timeout reached. Stop response: {stop_resp}')
            response.success = False
            response.message = 'UR timeout: program did not finish in time.'
            return response

        except Exception as e:
            response.success = False
            response.message = f'UR bridge exception: {e}'
            return response


def main():
    rclpy.init()
    node = URBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
