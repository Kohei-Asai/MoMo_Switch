import asyncio
from bleak import discover

async def run():
    devices = await discover()
    for d in devices:
        print(d)
loop = asyncio.new_event_loop()
try:
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run())
finally:
    asyncio.set_event_loop(None)
    loop.close()
#BLEのMACアドレス   B0B677EA-ACD8-4DC2-81B9-5AFE7D71F440