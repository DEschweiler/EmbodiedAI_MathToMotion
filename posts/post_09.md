---
id: 009
title: The Dog Gets a Brain — Agents and Tool Use
summary: Give an LLM tools and a control loop to act in the world.
tags: [agents, planning, tools]
learning_goals:
  - Describe the agent loop (perceive-plan-act-feedback).
  - Explain tool calls from an LLM and chain-of-thought planning.
  - Sketch the system architecture tying planner, skills, and sensors.
---

### From Text Generator to Agent

An LLM alone can't open a door or steer a robot. But it can *decide* which action to take next and call the appropriate tool.

The architecture: the LLM receives a system prompt listing available tools (e.g., `look()`, `move_forward(steps)`, `turn(direction)`). A user gives a natural language command. The LLM decomposes it into a sequence of tool calls, executes them one by one, receives feedback after each step, and adjusts its plan as needed.

### Chain-of-Thought Planning

For complex tasks ("find Prof. X and deliver document Y"), a single tool call isn't enough. Chain-of-thought prompting encourages the model to make planning steps explicit before acting: Where am I? → Plan route → Navigate → Verify arrival → Report. This dramatically improves plan quality compared to direct action generation.

### Reasoning

Modern LLMs can internally weigh options: "Should I go left or right around the chair? Left is tighter but shorter. Right is safer. → I'll go right." This internal deliberation is a key capability for embodied systems operating in unpredictable environments.

### The Full System Architecture

Bringing everything together:

  Voice command → LLM planner (cognitive layer)
    → Sequence of tool calls → Motor layer (ML skills: walk, turn, stop)
      → Physical execution (ROS2, motor currents)
        → Sensor feedback → back to LLM planner

Every post in this series has contributed a component. The LLM planner orchestrates them: data representations (Post 1), neural network training (Posts 2–3), foundation model adaptation (Post 4), vision (Post 5), language understanding (Posts 6–7), and multimodal perception (Post 8).

**ROS2 integration:** The LLM (running on a server) communicates with the robot (running its own hardware) via ROS2 — a messaging framework where components operate as nodes. The LLM is one node, motor control another, the camera a third.

### → Next

The system works on screen. But we've been hand-waving about one crucial component: how does the dog actually *walk*? We haven't trained a gait controller. That's fundamentally different from supervised learning — there are no "correct" walking data. The dog must discover how to walk through trial and error: reinforcement learning.
