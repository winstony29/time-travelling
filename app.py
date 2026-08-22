"""Time Travelling Stonks Man. POST /stonks."""
from flask import Flask, jsonify, request

from solver import solve

app = Flask(__name__)


@app.get("/")
def health():
    return jsonify(status="ok")


@app.post("/stonks")
def stonks():
    cases = request.get_json(force=True)
    return jsonify([solve(c) for c in cases])


if __name__ == "__main__":
    app.run(port=8000)
