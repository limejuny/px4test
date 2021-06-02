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

    print("-- 1.0m 상승")
    alt = 1.2
    interval = 0.4
    velocity = 2  # m/s
    (n, e) = (0, 0)
    await drone.offboard.set_position_ned(PositionNedYaw(0, 0.0, -alt, yaw))
    await asyncio.sleep(5)
    # lane인식 하기전까지 대기

    while True:
        with open('data.txt', 'r') as f:
            if f.read() != "stop":
                break

    while True:
        file = open('data.txt', 'r')
        s = file.read()
        # print(s)
        file.close()
        # print("loop")
        if s == "stop":
            break
        # 가끔 공백문자때문에 터져서 예외처리
        if s == '':
            s = 90.0
        deg = float(s) - 90
        yaw = (yaw + deg)  # % 360
        print(yaw)
        down = False
        async for position in drone.telemetry.position():
            down = position.absolute_altitude_m - init > alt
            # print(position.absolute_altitude_m)
            break
        # print(down)
        # print(((math.cos(math.pi * yaw / 180) * 3),
        #        (math.sin(math.pi * yaw / 180) * 3), 0.1 if down else -0.1, yaw))

        n += interval * velocity * math.cos(math.pi * yaw / 180)
        e += interval * velocity * math.sin(math.pi * yaw / 180)
        await drone.offboard.set_position_ned(
            PositionNedYaw(
                n,  # + interval * velocity * math.cos(math.pi * yaw / 180),
                e,  # + interval * velocity * math.sin(math.pi * yaw / 180),
                -alt,
                yaw))
        # await drone.offboard.set_velocity_ned(
        #     VelocityNedYaw((math.cos(math.pi * yaw / 180) * 3),
        #                    (math.sin(math.pi * yaw / 180) * 3),
        #                    0.1 if down else -0.1, yaw))

        # await drone.offboard.set_velocity_body(
        #     VelocityBodyYawspeed(3 * math.sin(math.pi * (yaw / 180)),
        #                          -3 * math.cos(math.pi * (yaw / 180)),
        #                          0.1 if down else -0.1, yaw - 90))
        # (yaw - 90) * math.pi / 180))
        # await drone.offboard.set_velocity_body(
        #     VelocityBodyYawspeed(3, 0, 0.1 if down else -0.1, 90 - yaw))
        await asyncio.sleep(interval)

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
