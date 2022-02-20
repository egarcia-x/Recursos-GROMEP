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

    return int.to_bytes(result, 8, "big")


class ArmMovement:
    __DISPLACEMENT_X_COMMAND = b'\x00'
    __DISPLACEMENT_Y_COMMAND = b'\x01'
    __DISPLACEMENT_Z_COMMAND = b'\x02'
    __GET_ARM_POSITION_COMMAND = b'\x03'
    __GRAB_PIECE = b'\x04'

    def __init__(self, usb_device: str):
        self.serial = Serial(usb_device, 9600, 8, 2, STOPBITS_ONE)

    def set_x(self, x: int):
        actual_x = self.get_x()
        desired_x = convert_to_a1_number(actual_x + x)

        message = bytearray(self.__DISPLACEMENT_X_COMMAND)
        message += bytearray(desired_x)

        self.serial.write(message)

    def set_y(self, y: int):
        actual_y = self.get_x()
        desired_y = convert_to_a1_number(actual_y + y)

        message = bytearray(self.__DISPLACEMENT_Y_COMMAND)
        message += bytearray(desired_y)

        self.serial.write(message)

    def set_z(self, z: int):
        actual_z = self.get_z()
        desired_z = convert_to_a1_number(actual_z + z)

        message = bytearray(self.__DISPLACEMENT_Z_COMMAND)
        message += bytearray(desired_z)

        self.serial.write(message)

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
        message += bytearray(b'\x00\x00')

        self.serial.write(message)

        response = bytearray(self.serial.read(112))

        message_x = convert_from_a1_number(response[16:47])
        message_y = convert_from_a1_number(response[48:79])
        message_z = convert_from_a1_number(response[80:112])

        return message_x, message_y, message_z

    def __del__(self):
        self.serial.close()
