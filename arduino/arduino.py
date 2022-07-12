import asyncio
from bleak import BleakClient
import datetime
# set characteristic uuid of Arduino, ESP32, etc...
CHARACTERISTIC_UUID = "59668694-8d7d-11eb-8dcd-0242ac130004"
# set ESP32 BLE address 
# ble_get_information.pyから取得して変更する
ADDRESS = "AE09B293-6D7E-FFD7-1697-EA9FCB160B8D"

class Esp32Ble():
    def __init__(self, data_dump_size: int=2048):
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
        print(
            "ACCX: {} ACCY: {} ACCZ: {}".format(self.accx, self.accy, self.accz) + "\n"
            + "GYRX: {} GYRY: {} GYRZ: {}".format(self.gyrx, self.gyry, self.gyrz) + "\n"
            + "MAGX: {} MAGY: {} MAGZ: {}".format(self.magx, self.magy, self.magz) + "\n"
        )
        now = datetime.datetime.now()
        print(now)
        print("\n")

    def notification_handler(self, sender, data):
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
    def stop(self):
        self.task.cancel()
                
    def main(self):
        self.loop = get_or_create_eventloop()
        self.task = asyncio.ensure_future(self.run(ADDRESS))
        self.loop.run_until_complete(self.task)
        # try:
        #     self.loop.run_until_complete(self.task)
        # except KeyboardInterrupt:
        #     print("KeyboardInterrupt!")
        #     self.task.cancel()
        # except self.stop():
        #     self.task.cancel()
        # finally:
        #     self.loop.close()
            
def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()
            
def ArduinoRun(n):
    global ble
    ble = Esp32Ble()
    ble.main()
            
if __name__ == "__main__":
    ArduinoRun(0)