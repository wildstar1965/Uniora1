from flask import Flask
import os

app = Flask(__name__)

@app.route("/")
def home():
    return "AI Backend Running 🚀"

@app.route("/run")
def run():
    return {"status": "working"}

# IMPORTANT PART
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render requires this
    app.run(host="0.0.0.0", port=port)