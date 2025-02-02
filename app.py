import cv2
from flask import Flask, Response, jsonify, render_template, request
from flask_mqtt import Mqtt
from ultralytics import YOLO
from utils.detector import PPEDetector
from utils.zone_monitor import ZoneMonitor

# write the flask app,output PPE_detection.py to index.html,replace video_feed_ppe.jpg with  PPE_detection.py output
