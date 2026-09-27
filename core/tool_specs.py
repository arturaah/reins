"""Reins tools offered to the chat model over MCP. Pure data: imported by the dashboard
(core/reins_tools.py implements them) and by the stdio MCP server (tools/reins_mcp.py).

None of these tools moves the physical robot. They observe, plan and preview in simulation;
physical motion requires human approval in the dashboard or paired glasses. Reins has no depth estimation (decision
2026-09-27): objects are seen in 2D only, so no tool locates or reaches for a real object.
"""

CAMERAS = ['head', 'left', 'right', 'glasses']
POSITION = {'type': 'array', 'minItems': 3, 'maxItems': 3, 'items': {'type': 'number', 'minimum': -2, 'maximum': 2},
            'description': 'metres in robot_base: x forward, y left, z up, origin on the floor under the pelvis'}

TOOL_SPECS = [
    {'name': 'get_robot_context',
     'description': ('What Reins can currently see and do: which cameras are online, the object detector, '
                     'the arms\' current simulated pose and reach geometry, and the planner/simulation state. '
                     'Call this first when unsure what is possible.'),
     'inputSchema': {'type': 'object', 'properties': {}, 'additionalProperties': False}},
    {'name': 'detect_objects',
     'description': ('Run the local object detector on the latest frame of one camera. Returns labels, '
                     'confidences and normalized image boxes [x0,y0,x1,y1] (0..1). These are 2D image regions, '
                     'not positions in metres. Pass labels to look for specific things by name (the default '
                     'detector is open-vocabulary, e.g. ["cup", "screwdriver"]); omit them for the default '
                     'household vocabulary. There is no depth: a box says where something is in the image, '
                     'not how far away it is.'),
     'inputSchema': {'type': 'object', 'properties': {
         'camera': {'type': 'string', 'enum': CAMERAS},
         'labels': {'type': 'array', 'items': {'type': 'string', 'minLength': 1, 'maxLength': 60}, 'maxItems': 30}},
         'required': ['camera'], 'additionalProperties': False}},
    {'name': 'plan_hand_path',
     'description': ('Plan a new single-arm motion from hand waypoints (gestures and compound sequences in the '
                     'space in front of the robot; never invented positions of real objects). The local planner solves IK, times the '
                     'motion and checks the whole path. Returns a proposal id and the validation result, or '
                     'the reason it was rejected so you can revise the waypoints. Operator approval is required for execution.'),
     'inputSchema': {'type': 'object', 'properties': {
         'name': {'type': 'string', 'minLength': 1, 'maxLength': 120},
         'arm': {'type': 'string', 'enum': ['left', 'right']},
         'waypoints': {'type': 'array', 'minItems': 1, 'maxItems': 16, 'items': {
             'type': 'object', 'properties': {'position_m': POSITION,
                                              'hold_s': {'type': 'number', 'minimum': 0, 'maximum': 5}},
             'required': ['position_m', 'hold_s'], 'additionalProperties': False}},
         'return_to_start': {'type': 'boolean'}},
         'required': ['name', 'arm', 'waypoints', 'return_to_start'], 'additionalProperties': False}},
    {'name': 'request_visual_guidance',
     'description': ('Request a task that needs camera context, such as a small non-contact approach to a visible object. '
                     'The primary planner runs first; the visual harness is the fallback. Never invent metric object positions. '
                     'Resulting steps use full-path validation and individual human approval. No execution or approval tool is exposed.'),
     'inputSchema': {'type': 'object', 'properties': {
         'task': {'type': 'string', 'minLength': 1, 'maxLength': 1000},
         'arm': {'type': 'string', 'enum': ['left', 'right']}},
         'required': ['task', 'arm'], 'additionalProperties': False}},
    {'name': 'preview_plan',
     'description': ('Play a validated proposal once in the dashboard\'s MuJoCo simulation so the user can '
                     'review it. This never moves the physical robot.'),
     'inputSchema': {'type': 'object', 'properties': {'proposal_id': {'type': 'string', 'minLength': 8, 'maxLength': 64}},
                     'required': ['proposal_id'], 'additionalProperties': False}},
]

# MCP tool annotations. Without them the spec assumes a tool may be destructive and reach the
# open world, and CLIs that cannot ask for approval (codex exec) hide such tools. Observation tools
# only read; planning/preview change only the dashboard's own proposal and simulation, never the robot.
_READ = {'readOnlyHint': True, 'destructiveHint': False, 'idempotentHint': True, 'openWorldHint': False}
_PLAN = {'readOnlyHint': False, 'destructiveHint': False, 'idempotentHint': False, 'openWorldHint': False}
for _spec in TOOL_SPECS:
    _spec['annotations'] = dict(_READ if _spec['name'] in ('get_robot_context', 'detect_objects') else _PLAN)

TOOL_NAMES = [t['name'] for t in TOOL_SPECS]

INSTRUCTIONS = """
You have Reins tools (MCP server "reins"). Use them instead of guessing or inventing geometry.
Never answer from the dashboard status summary what a camera shows: its detection list is usually
empty or stale. Call detect_objects to look.
- get_robot_context when unsure what is available (cameras, detector, reach).
- detect_objects to see what a camera shows. Boxes are 2D image regions, never positions in metres.
- plan_hand_path for a new gesture or arm motion in front of the robot; revise the waypoints from its
  rejection reason instead of relaxing constraints.
- preview_plan to show a validated proposal in the MuJoCo simulation.
Objects detected in 2D do not have measured metric positions. Never invent object coordinates.
For a real-object task, call request_visual_guidance to request camera-guided, non-contact steps.
The primary trajectory planner runs first; a bounded visual policy is used when it needs context.
Every generated step is validated, previewed and individually reviewed by the human.
No model tool approves or executes a plan. Only the operator can approve in the dashboard or paired
glasses after explicitly connecting robot control. Never claim execution from a successful planning call.
When you planned with a tool, set trajectory=null and robot_request=null
in your reply: the proposal is already in the dashboard's Motion preview.
"""
