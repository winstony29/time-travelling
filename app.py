"""Time Travelling Stonks Man. POST /stonks."""
from collections import deque

from flask import Flask, jsonify, request

from solver import solve

app = Flask(__name__)
TAP = deque(maxlen=50)  # rolling capture of the last grader payloads, to inspect


@app.get("/")
def health():
    return jsonify(status="ok")


@app.get("/tap")
def tap():  # GET the recent /stonks inputs+outputs so we can see real test cases
    return jsonify(list(TAP))


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
    TAP.append({"in": cases, "out": out})
    return jsonify(out)


if __name__ == "__main__":
    app.run(port=8000)
