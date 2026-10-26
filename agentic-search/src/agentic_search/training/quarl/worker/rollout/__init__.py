# The search agent loop now uses verl's stock ``ToolAgentLoop`` (registered
# as ``tool_agent`` by ``verl.experimental.agent_loop``) — no custom agent
# loop classes to register here anymore. Configure it with
# ``agent.default_agent_loop: tool_agent``.
#
# NOTE: sglang_rollout is intentionally NOT imported here — the ssp/kg
# (sglang) line is migrated separately and its module needs sglang 0.5.8.
