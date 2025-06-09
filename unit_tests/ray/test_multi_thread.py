import ray

ray.init(ignore_reinit_error=True)

@ray.remote
class ThreadedActor:
    def task_1(self):
        print("I'm running in a thread!")

    def task_2(self):
        print("I'm running in another thread!")

a = ThreadedActor.options(max_concurrency=2).remote()
ray.get([a.task_1.remote(), a.task_2.remote()])

ray.shutdown()