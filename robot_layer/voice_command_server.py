#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import json

from flask import Flask, request, jsonify

import rospy
from std_msgs.msg import String

app = Flask(__name__)
pub = None


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "node": "voice_command_server"}), 200


@app.route("/command", methods=["POST"])
def command():
    global pub

    try:
        data = request.get_json(force=True, silent=False)
    except Exception as e:
        return jsonify({"ok": False, "error": "invalid_json", "detail": str(e)}), 400

    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": "payload_must_be_object"}), 400

    intent = data.get("intent", None)
    if not intent:
        return jsonify({"ok": False, "error": "missing_intent"}), 400

    try:
        msg = json.dumps(data, ensure_ascii=False)
        pub.publish(msg)
        rospy.loginfo("Published /voice_cmd: %s", msg)
        return jsonify({"ok": True, "published": data}), 200
    except Exception as e:
        rospy.logerr("Failed to publish /voice_cmd: %s", str(e))
        return jsonify({"ok": False, "error": "publish_failed", "detail": str(e)}), 500


def main():
    global pub

    rospy.init_node("voice_command_server", anonymous=False)
    pub = rospy.Publisher("/voice_cmd", String, queue_size=10)

    host = rospy.get_param("~host", "0.0.0.0")
    port = int(rospy.get_param("~port", 5000))
    debug = bool(rospy.get_param("~debug", False))

    rospy.loginfo("voice_command_server listening on %s:%d", host, port)

    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    main()
