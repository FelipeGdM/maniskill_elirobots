from celery import Celery

app = Celery(
    "tarefas",
    broker="redis://172.17.0.3:6379/0",
    backend="redis://172.17.0.3:6379/0",  # necessário para guardar resultados
)

app.conf.update(
    result_expires=86400,  # equivalente ao result_ttl do RQ (1 dia)
    task_track_started=True,  # marca started explicitamente, não só pending/success
)
