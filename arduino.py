def get():
    import random
    return [random.uniform(-10,10),
            random.uniform(-10,10),
            random.uniform(-10,10),
            random.uniform(-10,10),
            random.uniform(-10,10),
            random.uniform(-10,10),
            random.uniform(-10,10),
            random.uniform(-10,10),
            random.uniform(-10,10)]

"""BLE connection between ESP32 and RaspberryPi"""
import asyncio
from bleak import BleakClient
import datetime
# set characteristic uuid of Arduino, ESP32, etc...
CHARACTERISTIC_UUID = "59668694-8d7d-11eb-8dcd-0242ac130004"
# set ESP32 BLE address 
# ADDRESS = "B0B677EA-ACD8-4DC2-81B9-5AFE7D71F440"
ADDRESS = "AE09B293-6D7E-FFD7-1697-EA9FCB160B8D"

class Esp32Ble(object):
    """Base model for BLE connect"""
    def __init__(self, data_dump_size: int=2048):
        """Initialize the value.
        """
        self.accx = 0
        self.accy = 0
        self.accz = 0
        self.gyrx = 0
        self.gyry = 0
        self.gyrz = 0
        self.magx = 0
        self.magy = 0
        self.magz = 0

    def value_print(self):
        """TX Value is get and print.
        ※ This program doesn't use the return value.
        Returns: 
            vrx (int): VRx axis value of joy stick
            vry (int): VRy axis value of joy stick
            sw  (int): sw ON/OFF of joy stick
        """
        print("ACCX: {} ACCY: {} ACCZ: {}".format(self.accx, self.accy, self.accz) + "\n" +"GYRX: {} GYRY: {} GYRZ: {}".format(self.gyrx, self.gyry, self.gyrz) + "\n" + "MAGX: {} MAGY: {} MAGZ: {}".format(self.magx, self.magy, self.magz) + "\n")
        now = datetime.datetime.now()
        print(now)
        print("\n")

    def notification_handler(self, sender, data):
        """Simple notification handler which prints the data received.
        
        Args:
            data (bytearray): bytearray data. For example b'***'
        """
        list_splitted = str(data).split(",")
        self.accx = list_splitted[0][12:]
        self.accy = list_splitted[1]
        self.accz = list_splitted[2]
        self.gyrx = list_splitted[3]
        self.gyry = list_splitted[4]
        self.gyrz = list_splitted[5]
        self.magx = list_splitted[6]
        self.magy = list_splitted[7]
        self.magz = list_splitted[8]
        self.value_print()
        
    async def run(self, address):
        """BLE asyncio loop function.
        Args:
            address (string): ESP32 address
        """
        async with BleakClient(address) as client:
            x = await client.is_connected()  # Check connection status
            print("Connected: {}".format(x))
            
            await client.start_notify(
                CHARACTERISTIC_UUID, self.notification_handler
            )
            while True:
                try:
                    await asyncio.sleep(1.0)                   
                except Exception:
                    break
                
    def main(self):
        """BLE main function.
        """
        loop = asyncio.get_event_loop()
        task = asyncio.ensure_future(self.run(ADDRESS))
        try:
            loop.run_until_complete(task)
        except KeyboardInterrupt:
            print("KeyboardInterrupt!")
            task.cancel()
            # loop.run_forever()
        finally:
            loop.close()
            
if __name__ == "__main__":
    ble = Esp32Ble()
    ble.main()