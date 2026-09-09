from flask import Flask, jsonify, render_template_string

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>SpotMicro Controller</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: Arial;
            text-align: center;
            background: #111;
            color: white;
        }
        button {
            width: 120px;
            height: 60px;
            margin: 8px;
            font-size: 20px;
            border-radius: 12px;
        }
        .stop {
            background: red;
            color: white;
        }
    </style>
</head>
<body>
    <h1>🐕 SpotMicro</h1>
    <p id="status">Ready</p>

    <div>
        <button onclick="command('forward')">Forward</button>
    </div>
    <div>
        <button onclick="command('left')">Left</button>
        <button class="stop" onclick="command('stop')">STOP</button>
        <button onclick="command('right')">Right</button>
    </div>
    <div>
        <button onclick="command('backward')">Backward</button>
    </div>

    <script>
        function command(cmd) {
            fetch('/command/' + cmd)
                .then(response => response.json())
                .then(data => {
                    document.getElementById('status').innerText =
                        'Command: ' + data.command;
                });
        }
    </script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML)

@app.route("/command/<cmd>")
def command(cmd):
    allowed = ["forward", "backward", "left", "right", "stop"]

    if cmd not in allowed:
        return jsonify({"error": "Invalid command"}), 400

    print("COMMAND:", cmd)

    return jsonify({
        "status": "ok",
        "command": cmd
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
