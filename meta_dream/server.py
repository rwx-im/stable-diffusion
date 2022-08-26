#!/usr/bin/env python

from flask import Flask, request
from rq import Queue, Worker
from rq.job import Job
from redis import Redis
import rq_dashboard

from stable_diffusion import do_work
from prompt import Prompt
from upload import upload_image
import config

app = Flask(__name__)
app.config.from_object(rq_dashboard.default_settings)
app.config[
    "RQ_DASHBOARD_REDIS_URL"
] = f"redis://nobody:{config.REDIS_PASSWORD}@{config.REDIS_HOST}"
app.register_blueprint(rq_dashboard.blueprint, url_prefix="/rq")
redis = Redis(
    host=config.REDIS_HOST,
    password=config.REDIS_PASSWORD,
)
queue = Queue(connection=redis)
upload_queue = Queue("uploads", connection=redis)


def valid_dream(json):
    if not "prompt" in json:
        return False

    return True


@app.route("/")
def index():
    return ""


def job_not_found(job_id):
    return ({"error": "the job was not found"}, 404)


def job_to_dict(job):
    return {
        "id": job.id,
        "status": job.get_status(),
        "result": job.result,
        "enqueued_at": job.enqueued_at,
        "started_at": job.started_at,
        "ended_at": job.ended_at,
    }


def worker_to_dict(worker):
    return {
        "name": worker.name,
        "hostname": worker.hostname,
        "pid": worker.pid,
        "queues": [queue.name for queue in worker.queues],
        "state": worker.state,
        "current_job_id": worker.get_current_job_id(),
        "last_heartbeat": worker.last_heartbeat,
        "birth_date": worker.birth_date,
        "successful_job_count": worker.successful_job_count,
        "failed_job_count": worker.failed_job_count,
        "total_working_time": worker.total_working_time,
    }


@app.route("/api/v1/jobs")
def list_jobs():
    jobs = queue.get_jobs()

    return ([job_to_dict(job) for job in jobs], 200)


@app.route("/api/v1/jobs/finished")
def list_finished_jobs():
    registry = queue.finished_job_registry

    return (registry.get_job_ids(), 200)


@app.route("/api/v1/jobs/started")
def list_started_jobs():
    registry = queue.started_job_registry

    return (registry.get_job_ids(), 200)


@app.route("/api/v1/jobs/failed")
def list_failed_jobs():
    registry = queue.failed_job_registry

    return (registry.get_job_ids(), 200)


@app.route("/api/v1/jobs/canceled")
def list_canceled_jobs():
    registry = queue.canceled_job_registry

    return (registry.get_job_ids(), 200)


@app.route("/api/v1/jobs/<job_id>")
def get_job(job_id):
    try:
        job = Job.fetch(job_id, connection=redis)

        return (job_to_dict(job), 200)
    except rq.exceptions.NoSuchJobError:
        return job_not_found(job_id)


@app.route("/api/v1/workers")
def list_workers():
    workers = Worker.all(connection=redis)

    return ([worker_to_dict(worker) for worker in workers], 200)


@app.route("/api/v1/dream", methods=["POST"])
def post_dream():
    if not request.is_json:
        return ({"message": "invalid json"}, 400)

    json = request.get_json()

    print(json)

    if valid_dream(json):
        prompt = Prompt(text=json["prompt"], seed=int(json.get("seed", 1)))
        dream_job = queue.enqueue(do_work, prompt, result_ttl=3600)
        queue_position = dream_job.get_position()

        return ({"job": dream_job.id, "queue_position": queue_position}, 201)

    return ({"error": "invalid request"}, 422)


if __name__ == "__main__":
    app.run(host="0.0.0.0")
