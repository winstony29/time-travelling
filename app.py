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
    # never let one hard/broken case 500 the whole batch: worst case is []
    out = []
    for c in cases:
        try:
            out.append(solve(c))
        except Exception:
            out.append([])
    return jsonify(out)


if __name__ == "__main__":
    app.run(port=8000)
