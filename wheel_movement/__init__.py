import requests

class WheelMovement:
    def __init__(self, server_ip: str, server_port: int):
        self.server_ip = server_ip
        self.server_port = server_port

    def send_x_displacement(displacement_x: int): 
        json_body = {
            "displacement_x": displacement_x
        }

        response = requests.post("http://" + server_ip + ":" + server_port + "/positions")

        if response.status_code == 200:
            return
        else:
            raise RuntimeError("Error processing the new position")


    def send_y_displacement(displacement_y: int): 
        json_body = {
            "displacement_y": displacement_y
        }

        response = requests.post("http://" + server_ip + ":" + server_port + "/positions", json_body)

        if response.status_code == 200:
            return
        else:
            raise RuntimeError("Error processing the new position")

    def send_x_y_displacement(displacement_x: int, displacement_y: int):
        json_body = {
            "displacement_x": displacement_x,
            "displacement_y": displacement_y
        }

        response = requests.post("http://" + server_ip + ":" + server_port + "/positions", json_body)

        if response.status_code == 200:
            return
        else:
            raise RuntimeError("Error processing the new position")