import socket
import sys
import signal
import json
import struct
import numpy as np
import matplotlib.pyplot as plt

from rc import RadarConfig
from dsp import RadarDSP

global xvals
global yvals
first_config = True

local_ip = socket.gethostbyname(socket.gethostname())
PORT = 4242

def signal_handler(signal, frame):
    # close the socket here
    sys.exit(0)

def process_config(json_config):
    global first_config
    global radar_config
    global radar_dsp

    config = json_config["config"]

    if (first_config):
        radar_config = RadarConfig(config)
        radar_dsp = RadarDSP(radar_config)
    else:
        radar_config.update_vals(config)

    first_config = False

def process_data(json_data):
    radar_dsp.process_packet(json_data)

signal.signal(signal.SIGINT, signal_handler)

server_address = (local_ip, PORT)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

sock.bind(server_address)

sock.listen(1)

print("Starting server on: " + str(server_address))
def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def recv_msg(sock):
    # Listen for length header
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None

    msglen = int.from_bytes(raw_msglen, byteorder='little', signed=False)

    # Read the rest of the packet
    return recvall(sock, msglen)

while True:
    try:
        connection, address = sock.accept()
        print ("Connected from " + str(address))

        while True:
            msg = recv_msg(connection)
            if not msg:
                break

            msg = json.loads(msg)

            if (msg["packet_type"] == "configuration"):
                process_config(msg)
            else:
                process_data(msg)

    except KeyboardInterrupt:
        break

sock.shutdown()
sock.close()
