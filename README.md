# JetBot Voice-to-Action Navigation

This repository contains the source code for the final project:

**Perancangan Voice-to-Action Berbasis LLM pada JetBot untuk Navigasi Indoor dengan LiDAR dan IMU**

Author: Yuki Resky Damanik  
Student ID: 23402210008  
Program: Rekayasa Sistem Komputer, Universitas Prasetiya Mulya

## Overview

This project implements a voice-to-action pipeline for indoor navigation on a JetBot robot. The system converts voice commands into structured JSON commands using ASR and an LLM parser, validates them using a safety guard, and sends valid commands to the JetBot robot layer for ROS-based navigation.

## System Layers

### Voice Layer

The voice layer runs on a laptop and handles:

- Speech-to-text using RealtimeSTT/Whisper
- LLM-based intent parsing
- Safety guard validation
- JSONL logging
- HTTP command sending to JetBot

### Robot Layer

The robot layer runs on JetBot and handles:

- HTTP command server
- ROS topic `/voice_cmd`
- Command executor
- POI-based navigation
- LiDAR, IMU, map, localization, and motor control
- `/cmd_vel` motor command bridge

## Repository Structure

```text
voice_layer/      Voice processing, LLM parser, and safety guard
robot_layer/      JetBot ROS command server, executor, and motor bridge
config/           POI and configuration files
gazebo_sim/       Gazebo simulation code and notes
logs_sample/      Example log format
docs/             Diagrams and supporting documentation
