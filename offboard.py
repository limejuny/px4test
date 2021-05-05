#!/usr/bin/env python3

import asyncio

from mavsdk import System
from mavsdk.offboard import (OffboardError, PositionNedYaw, VelocityNedYaw)


async def run():
    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("-- Arming")
    await drone.action.arm()

    print("-- Setting initial setpoint")
    await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, 0.0, 0.0))

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

    print("-- 5m 상승")
    await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, -5.0, 90.0))
    await asyncio.sleep(8)

    print("-- 동쪽으로 20m")
    await drone.offboard.set_position_ned(PositionNedYaw(0.0, 20.0, -5.0, 90.0))
    await asyncio.sleep(8)

    print("-- 왼쪽 2m/s 위쪽 1.5m/s로 10초간 이동 (25m이동, x:0 y:15)")  # arccos(3/5)
    await drone.offboard.set_velocity_ned(VelocityNedYaw(
        1.5, -2.0, 0.0, -53.13))
    await asyncio.sleep(10)

    print("-- 밑으로 5m")
    # await drone.offboard.set_position_ned(PositionNedYaw(10.0, 0.0, -5.0, 45.0))
    await drone.offboard.set_velocity_ned(VelocityNedYaw(
        -1.0, 0.0, 0.0, -180.0))
    await asyncio.sleep(5)

    print("-- Stopping offboard")
    try:
        await drone.offboard.stop()
    except OffboardError as error:
        print(
            f"Stopping offboard mode failed with error code: {error._result.result}"
        )
    await drone.action.disarm()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())
