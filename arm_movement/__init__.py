from serial import Serial, STOPBITS_ONE


def convert_from_a1_number(a1_number: bytes) -> int:
    result = 0
    hex_to_bin = format(int.from_bytes(a1_number, "big"), '0>32b')

    if int.from_bytes(a1_number, "big") > 4294967295:
        raise ArithmeticError("The numbers is outside the range [-2147483647, 2147483647]")

    if hex_to_bin[0] == '1':
        iterator = len(hex_to_bin)

        for bit in hex_to_bin:
            iterator -= 1
            result += (1 - int(bit)) * pow(2, iterator)

        result *= -1
    else:
        result = int.from_bytes(a1_number, "big")

    return result


def convert_to_a1_number(number: int) -> bytes:
    result = 0
    abs_number = abs(number)

    if number < -2147483647 or 2147483647 < number:
        raise ArithmeticError("The numbers is outside the range [-2147483647, 2147483647]")

    if number < 0:
        int_to_bin = format(abs_number, '0>32b')

        iterator = len(int_to_bin) - 1

        for bit in int_to_bin:
            result += (1 - int(bit)) << iterator
            iterator -= 1
    else:
        result = number

    return int.to_bytes(result, 4, "big")


class ArmMovement:
    __DISPLACEMENT_X_COMMAND = b'\x00'
    __DISPLACEMENT_Y_COMMAND = b'\x01'
    __DISPLACEMENT_Z_COMMAND = b'\x02'
    __GET_ARM_POSITION_COMMAND = b'\x03'
    __GRAB_PIECE = b'\x04'

    def __init__(self, usb_device: str):
        self.serial = Serial(usb_device, 9600, 8, 2, STOPBITS_ONE)

    def set_x(self, x: int):
        self.__send_new_position(self.__DISPLACEMENT_X_COMMAND, self.get_x(), x)

    def set_y(self, y: int):
        self.__send_new_position(self.__DISPLACEMENT_Y_COMMAND, self.get_y(), y)

    def set_z(self, z: int):
        self.__send_new_position(self.__DISPLACEMENT_Z_COMMAND, self.get_z(), z)

    def set_position(self, position: (int, int, int)):
        self.set_x(position[0])
        self.set_y(position[1])
        self.set_z(position[2])

    def get_x(self) -> int:
        return self.get_position()[0]

    def get_y(self) -> int:
        return self.get_position()[1]

    def get_z(self) -> int:
        return self.get_position()[2]

    def get_position(self) -> (int, int, int):
        message = bytearray(self.__GET_ARM_POSITION_COMMAND)
        message.append(b'\x00\x00\x00\x00')

        self.serial.write(message)

        return self.__wait_result_response()

    def grab_piece(self):
        message = bytearray(self.__GRAB_PIECE)
        message.append(b'\x00\x00\x00\x00')

        self.serial.write(message)
        self.__wait_status_response()

    def __send_new_position(self, command: __DISPLACEMENT_X_COMMAND | __DISPLACEMENT_Y_COMMAND | __DISPLACEMENT_Z_COMMAND, actual_axis: int, new_axis: int):
        desired_pos = convert_to_a1_number(actual_axis + new_axis)

        message = bytearray(command)
        message.append(desired_pos)

        self.serial.write(message)
        self.__wait_status_response()

    def __wait_status_response(self):
        response = bytearray(self.serial.read(2))

        result_type = response[0:1]
        status_code = response[2:3]

        if result_type != b'\x00\x01':
            raise RuntimeError("Receive unexpected response")
        elif status_code == b'\x00\x01':
            raise RuntimeError("The Serial Device raise error during processing the command")

    def __wait_result_response(self) -> (int, int, int):
        response = bytearray(self.serial.read(14))

        result_type = response[0:1]
        message_x = convert_from_a1_number(int.to_bytes(response[2:5], 4, "big"))
        message_y = convert_from_a1_number(int.to_bytes(response[6:9], 4, "big"))
        message_z = convert_from_a1_number(int.to_bytes(response[10:13], 4, "big"))

        if result_type != b'\x00\x00':
            raise RuntimeError("Receive unexpected response")

        return message_x, message_y, message_z

    def __del__(self):
        self.serial.close()
