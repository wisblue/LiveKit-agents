import asyncio

async def define_progress():
    print("progress started")
    await asyncio.sleep(2)
    print("progress ended")

    for i in range(10):
        print(i)
        await asyncio.sleep(0.1)

    n = 10
    while True:
        n -= 1
        if n < 0:
            break
        print(n)
        await asyncio.sleep(0.1)


asyncio.run(define_progress())
