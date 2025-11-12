from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def home():
    return "<h1>Loan Mountain Capital Estimator</h1><p>Beta version running!</p>"
