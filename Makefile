api:
	PYTHONPATH=src uvicorn video_agent.api:app --reload --host 0.0.0.0 --port 8000

test:
	PYTHONPATH=src pytest tests -v

eval:
	PYTHONPATH=src python -m video_agent.evaluate