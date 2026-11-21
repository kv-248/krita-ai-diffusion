import asyncio
import gc
import weakref

from ai_diffusion import eventloop


def test_run_retains_suspended_task():
    """Fire-and-forget tasks must survive GC while waiting on a future.

    asyncio only keeps a weak reference to tasks. Callers such as server install
    drop the object returned by eventloop.run, so without an internal strong
    reference a collection destroys the task and injects GeneratorExit.
    """
    log: list[str] = []

    async def suspended():
        future = asyncio.get_running_loop().create_future()
        try:
            await future
            log.append("finished")
        except BaseException as e:
            log.append(type(e).__name__)
            raise

    task = eventloop.run(suspended())
    eventloop.process_python_events()
    assert not task.done()

    ref = weakref.ref(task)
    del task
    gc.collect()

    retained = ref()
    assert retained is not None
    assert log == []

    retained.cancel()
    # Cancellation finishes the task on the first turn and queues the done
    # callback; the next turn drops the strong reference.
    eventloop.process_python_events()
    eventloop.process_python_events()
    del retained
    gc.collect()
    assert ref() is None
