connect:
	ssh root@159.89.55.72 -i ~/.ssh/id_ed25519

run:
	python3 -m uvicorn app:app --host 0.0.0.0 --port 8000
run-pm2:
	pm2 start "myenv/bin/uvicorn app:app --host 0.0.0.0 --port 8000" --name fastapi_app