import pyaudio
import socket
import numpy as np
import time
import threading
# Thiết lập thông số kết nối
HOST = '127.0.0.1'
PORT = 12345
# Thiết lập các thông số ghi âm
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
# Thiết lập ngưỡng âm thanh
THRESHOLD = 3
# Hàm gửi âm thanh đến client
def send_audio_data(client_socket):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    while True:
        data = stream.read(CHUNK)
        audio_data = np.frombuffer(data, dtype=np.int16)
        if np.max(audio_data) > THRESHOLD:
            print("Started recording...")
            while np.max(audio_data) > THRESHOLD:
                client_socket.sendall(data)
                data = stream.read(CHUNK)
                audio_data = np.frombuffer(data, dtype=np.int16)
            print("Stopped recording.")
        time.sleep(1)
    stream.stop_stream()
    stream.close()
    p.terminate()
# Thiết lập kết nối và gửi dữ liệu
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    server_socket.bind((HOST, PORT))
    server_socket.listen()
    print("Server is listening...")
    client_conn, client_addr = server_socket.accept()
    print(f"Client connected: {client_addr}")
    send_thread = threading.Thread(target=send_audio_data, args=(client_conn,))
    send_thread.start()
