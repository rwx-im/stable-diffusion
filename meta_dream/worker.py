#!/usr/bin/env python

from redis import Redis
from rq import Connection, SimpleWorker


import stable_diffusion
import config
from upload import upload_image


def main():
    with Connection(
        connection=Redis(
            host=config.REDIS_HOST,
            password=config.REDIS_PASSWORD,
        )
    ):
        w = SimpleWorker("default")
        w.work()


if __name__ == "__main__":
    main()
