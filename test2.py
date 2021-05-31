#!/usr/bin/env python3

import asyncio
import math

from mavsdk import System, telemetry
from mavsdk.offboard import (OffboardError, PositionNedYaw, VelocityNedYaw,
                             VelocityBodyYawspeed)


async def run():
    file = open('data.txt', 'w')
    file.write('stop')
    file.close()
    drone = System()
    await drone.connect(system_address="udp://:14540")

    init = 0
    async for position in drone.telemetry.position():
        init = position.absolute_altitude_m
        break
    print("-- Arming")
    try:
        await drone.action.arm()
    except Exception as e:
        print(e)

    print("-- Setting initial setpoint")
    await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, 0.0, 90.0))
    yaw = 90.0
    print("-- Starting offboard")
    try:
        await drone.offboard.start()
    except OffboardError as error:
        print(
            f"Starting offboard mode failed with error code: {error._result.result}"
        )
        print("-- Disarming")
        await drone.action.disarm()
        return

    print("-- 1.5m 상승")
    await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, -1.5, yaw))
    await asyncio.sleep(10)
    while True:
        file = open('data.txt', 'r')
        s = file.read()
        # print(s)
        file.close()
        # print("loop")
        if s == "stop":
            break
        else:
            deg = 90.0 - float(s)
            yaw += deg
            # print(yaw)
            down = False
            async for position in drone.telemetry.position():
                down = position.absolute_altitude_m - init > 1.5
                # print(position.absolute_altitude_m)
                break
            # print(down)
            await drone.offboard.set_velocity_ned(
                VelocityNedYaw((math.sin(math.pi * ((yaw + 90) / 180)) * 3),
                               (math.cos(math.pi * ((yaw - 90) / 180)) * 3),
                               0.1 if down else -0.1, yaw))
            # await drone.offboard.set_velocity_body(
            #     VelocityBodyYawspeed(3, 0, 0.1 if down else -0.1, yaw))
            await asyncio.sleep(0.2)

    print("-- Stopping offboard")
    try:
        await drone.offboard.stop()
    except OffboardError as error:
        print(
            f"Stopping offboard mode failed with error code: {error._result.result}"
        )

    await asyncio.sleep(5)

    await drone.action.land()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())
