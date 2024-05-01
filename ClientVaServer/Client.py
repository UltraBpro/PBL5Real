import socket
import pyaudio
import threading
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('localhost', 12345))
def receive_and_play():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    output=True,
                    frames_per_buffer=CHUNK)
    while True:
        try:
            data = client_socket.recv(CHUNK)
            if not data:
                break
            stream.write(data)
        except socket.error as err:
            print(f"Error receiving data: {err}")
            break
    stream.stop_stream()
    stream.close()
    p.terminate()
    client_socket.close()
receive_thread = threading.Thread(target=receive_and_play)
receive_thread.start()
