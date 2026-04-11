---
id: 009
title: The Dog Gets a Brain — Agents and Tool Use
summary: Give an LLM tools and a control loop to act in the world.
tags: [agents, planning, tools]
learning_goals:
  - Describe the agent loop (perceive-plan-act-feedback) and its components.
  - Explain how an LLM generates tool calls and processes their results.
  - Understand chain-of-thought and ReAct-style reasoning for planning.
  - Sketch the full system architecture connecting planner, motor skills, and sensors.
---

Sessions 1–8 built every component separately: neural networks, vision, language, multimodal perception. Now they converge. A single natural language command — "go to the door and wait" — triggers a cascade that touches all of them. The question this session answers: what sits at the top, orchestrating everything?

The answer is an **agent**: an LLM equipped with tools, a memory of past steps, and a loop that keeps it running until the task is done.

## What Is an Agent?

An agent is a system that perceives its environment, decides on an action, executes it, observes the result, and repeats. This perceive-plan-act-observe loop is not new — it is the basic architecture of every control system. What's new is placing a large language model at the center of the loop as the planner, with learned sub-skills available as callable tools.

Formally, an agent operates in a loop:

```
while task_not_complete:
    observation = perceive()          # camera, sensors, tool results
    thought = llm(context + observation)  # plan next step
    action = extract_tool_call(thought)   # parse the intended action
    result = execute(action)          # call the tool
    context.append(observation, thought, action, result)
```

The context grows with each iteration: the LLM always has access to the full history of what it has seen, planned, and done. This is what allows it to adapt when something unexpected happens.

## Tool Calls

The mechanism that connects the LLM to the real world is **tool use**. The LLM is given a system prompt that lists available tools with their signatures and descriptions:

```
Available tools:
- look(direction: str) → str: capture camera image, describe what is visible
- move_forward(meters: float) → str: walk forward the specified distance
- turn(degrees: float) → str: rotate in place, positive = clockwise
- speak(text: str): output speech through onboard speakers
- get_position() → dict: return current estimated position {x, y, heading}
```

When the LLM decides to use a tool, it emits a structured call in its output — typically a JSON block or a function call in a prescribed format:

```json
{
  "tool": "look",
  "arguments": {"direction": "forward"}
}
```

The agent runtime intercepts this call, executes it, and returns the result as the next observation in the context. The LLM sees:

```
Tool result: "I see a corridor. There is a closed door at the end, approximately 8 meters ahead. A chair is positioned 2 meters to the left."
```

And continues planning.

### Why Not Just Output Actions Directly?

One alternative: train a neural network to map observations directly to motor commands, bypassing language entirely. This is what the motor layer does (Sessions 10–11). But direct action policies are brittle: they handle what they were trained for and fail quietly on anything else.

The LLM agent adds a layer of flexible reasoning: it can handle novel situations by combining tools in new ways, ask for clarification when uncertain, and explain its decisions in language. The trade-off is latency — an LLM inference call takes time — which is why the cognitive and motor layers are kept separate.

## Chain-of-Thought and ReAct

A naive agent that immediately emits tool calls often makes poor decisions on complex tasks. **Chain-of-thought prompting** (Session 7) asks the model to make its reasoning explicit before acting:

```
Thought: The user wants me to find Prof. Müller's office and deliver the report.
I don't know where Prof. Müller's office is. I should first ask the robot's map
or look for a directory sign.
Action: look(direction="forward")
```

**ReAct** (Reasoning + Acting; Yao et al., 2022) formalizes this pattern: every agent step consists of a *thought* (free-text reasoning) followed by an *action* (tool call). The model alternates between reasoning about what it knows and acting to gather more information. This dramatically improves performance on tasks requiring multi-step information gathering.

The same pattern applies to the robot:

```
Thought: I need to navigate to the laboratory on the second floor.
I am currently on the first floor. I need to find the staircase.
Action: look(direction="forward")
Observation: "Corridor, no staircase visible."
Thought: The staircase might be around the corner. I'll turn left.
Action: turn(degrees=-90)
Action: move_forward(meters=3)
Action: look(direction="forward")
Observation: "Staircase visible, approximately 4 meters ahead."
Thought: Found the staircase. I'll approach it.
Action: move_forward(meters=4)
```

Each thought-action-observation triple is one iteration of the loop.

## Memory and Context Management

The agent's context window fills up over a long task. Several strategies manage this:

**Sliding window:** keep only the $k$ most recent steps in context; discard older ones. Simple but risks losing important earlier information.

**Summarization:** periodically ask the LLM to summarize the task history so far, replacing raw steps with a compressed summary. Preserves task state while reducing token count.

**External memory:** store observations and plans in a structured external database (e.g., a key-value store or vector database). The LLM retrieves relevant entries by query rather than reading the full history. This is the approach used in more sophisticated agent frameworks.

For a short navigation task, a sliding window of the last 10 steps is usually sufficient. For long-horizon tasks (exploring an entire building), external memory becomes necessary.

## Error Handling and Replanning

Real environments produce unexpected outcomes. The motor skill "move_forward(2)" might fail because a person stepped in front of the robot. A robust agent must detect failures and replan:

```
Action: move_forward(meters=2)
Observation: "Path blocked by a person. Movement aborted after 0.3 meters."
Thought: I cannot proceed forward. I should wait for the person to pass,
or find an alternative route around them.
Action: speak(text="Excuse me, I need to pass through.")
```

This error-handling capability emerges from the LLM's general reasoning: it doesn't need explicit "if path blocked then..." rules. It reasons from the observation just as it would reason about any new information.

## The Full System Architecture

Bringing all sessions together, the complete Unitree Go2 agent stack:

```
Natural language command (voice/text)
    ↓
Speech recognition (Whisper or similar)
    ↓
LLM Planner — cognitive layer (Session 9)
  ├── Tool: look() → VLM perception (Sessions 5, 8)
  ├── Tool: move_forward() → locomotion policy (Session 10)
  ├── Tool: turn() → locomotion policy (Session 10)
  ├── Tool: get_position() → sensor fusion / SLAM
  └── Tool: speak() → text-to-speech
    ↓
Motor commands → ROS2 middleware
    ↓
Physical hardware — joint controllers (Session 11)
    ↓
Sensor readings → back to planner loop
```

Every session has contributed a component:
- Sessions 1–3: the mathematical and training foundations for every learned model in this stack.
- Session 4: self-supervised pretraining that gives the vision and language components their representational power.
- Session 5: the vision encoder that processes camera images into patch embeddings.
- Session 6: the language model that parses commands and generates plans.
- Session 7: instruction tuning and RLHF that make the LLM follow directives reliably.
- Session 8: multimodal fusion via CLIP and VLMs that connects vision to language.
- Session 9: the agent loop and tool use that tie everything into a working system.
- Sessions 10–11: locomotion learning and sim-to-real transfer for the motor layer.

### ROS2 Integration

The LLM planner typically runs on a server (cloud or onboard GPU). The robot hardware runs ROS2 — a publish-subscribe messaging framework where each functional component is a node. Nodes communicate via topics (streams of sensor data) and services (request-response calls).

The agent runtime is a ROS2 node that: receives sensor data (camera images, IMU, joint states), calls the LLM API, parses the tool call from the response, dispatches the call to the appropriate ROS2 service or topic, and publishes the result back to the LLM context.

This architecture cleanly separates the cognitive layer (LLM, runs on GPU server) from the physical layer (motor controllers, run on embedded hardware), connected by a well-defined message interface.

---

## Further Reading

**Start here** *(accessible introductions)*
- Lilian Weng: "LLM-powered Autonomous Agents" — [lilianweng.github.io/posts/2023-06-23-agent](https://lilianweng.github.io/posts/2023-06-23-agent) — the definitive survey blog post on agent architectures; clear, comprehensive, and well-illustrated
- Harrison Chase: LangChain conceptual documentation on agents — [python.langchain.com/docs/concepts/agents](https://python.langchain.com/docs/concepts/agents) — practical framework-level explanation of agent loops, tools, and memory
- Andrej Karpathy: "State of GPT" (Microsoft Build 2023) — [youtube.com/watch?v=bZQun8Y4L2A](https://www.youtube.com/watch?v=bZQun8Y4L2A) — excellent overview of how LLMs are used as system components, including agents

**Go deeper** *(technical references)*
- Yao et al.: "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023) — [arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629) — the ReAct framework for interleaving chain-of-thought and tool calls
- Schick et al.: "Toolformer: Language Models Can Teach Themselves to Use Tools" (NeurIPS 2023) — [arxiv.org/abs/2302.04761](https://arxiv.org/abs/2302.04761) — how LLMs can learn tool use from self-generated training data
- Driess et al.: "PaLM-E: An Embodied Multimodal Language Model" (ICML 2023) — [arxiv.org/abs/2303.03378](https://arxiv.org/abs/2303.03378) — large-scale embodied LLM agent applied to robot manipulation and navigation
- Brohan et al.: "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control" (2023) — [arxiv.org/abs/2307.15818](https://arxiv.org/abs/2307.15818) — end-to-end approach where the VLM directly outputs robot actions

---

## ✏️ Check Your Understanding *(Preliminary)*

> *These questions are placeholders to be refined. Use them to self-assess before moving on.*

1. The perceive-plan-act-observe loop is the core of the agent architecture. What information is passed at each step? Draw the loop and annotate each arrow with the type of data flowing through it.

2. An agent is given the command "bring me a water bottle from the kitchen." List at least five tool calls the agent might need to make, in order, including what results it expects from each call. Where might the plan need to deviate from the expected sequence?

3. Chain-of-thought (Session 7) and the ReAct pattern both involve generating explicit reasoning steps. What does ReAct add beyond standard chain-of-thought? Why is it particularly useful for embodied agents?

4. The session distinguishes the cognitive layer (LLM planner) from the motor layer (locomotion policy). Why is it important to keep these separate rather than training a single model that maps language commands directly to motor currents?

5. *(Preliminary)* Consider the context management problem: a long exploration task fills the agent's context window. What are the trade-offs between the three strategies presented (sliding window, summarization, external memory)? Which would you choose for a task that takes 20 minutes of continuous operation?

---

### → Next

The system works on screen. But we've been hand-waving about one crucial component: how does the dog actually *walk*? We haven't trained a gait controller. That's fundamentally different from supervised learning — there are no "correct" walking demonstrations to learn from. The dog must discover effective movement strategies through trial and error: reinforcement learning. Session 10 covers the mathematical foundations and practical algorithms that make this possible.
