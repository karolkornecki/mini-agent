# mini-agent

The language used for the coding agent's tool execution and context monitoring.

## Language

**Tool call**:
A model's request to run a named tool with supplied arguments and a call identifier.

**Tool execution**:
An attempt to carry out a tool call, including checking its arguments and permission
and obtaining approval when required.

**Tool result**:
A response to the model for a particular tool call, containing its output or the
reason it could not be carried out.

**Tool error**:
A failure to fulfill a tool call because of an invalid request, policy denial, or
handled execution failure, including a command timeout. An approval rejection is
not a tool error.

**Approval rejection**:
The user's refusal to allow a tool call that requires approval.
_Avoid_: Tool error

**Context utilization**:
The estimated percentage of configured context capacity used by the input to the
most recently completed model request, based on provider-reported token usage.
It is unknown when that request has no usage data.
