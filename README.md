# Gemini Web Search agent tool

This Dataiku plugin provides a custom agent tool that calls Gemini on Vertex AI with
Google Search grounding. It searches the live public web; it does **not** require or
query a Vertex AI Search datastore/application. It returns Gemini's answer, the web
search queries it generated, and the referenced source URLs.

## Configure

1. Install the plugin and create an **Agent Tool** of type **Gemini Web Search**.
2. Select a Dataiku Vertex AI LLM connection configured with either `OAUTH` or `KEYPAIR`
   (Private Key) authentication. The user who invokes the tool must be allowed to use
   that connection.
3. The tool uses the connection `projectId` and `location`/`region` by default. Set
   either override only when required, then choose a Gemini model that supports
   Google Search grounding.
4. Add the tool to a visual agent and describe when its knowledge base should be used.

### Authentication

The tool supports two authentication methods:

- **OAuth**: Uses the short-lived OAuth token resolved from the Dataiku connection at
  invocation time. No credential is stored in the tool configuration or returned to the
  agent.
- **Private Key (KEYPAIR)**: Uses a service account JSON key configured in the Dataiku
  connection. The service account must have the `https://www.googleapis.com/auth/cloud-platform`
  scope enabled.

The Google identity (OAuth user or service account) configured on the connection needs
permission to access the specified Vertex AI service. Vertex AI and the Google Search
grounding feature must be enabled for the chosen project, model, and region.

## Test payload

```json
{"input": {"query": "What are the latest James Webb Space Telescope updates?"}, "context": {}}
```

The tool is stateless: each call is a new grounded Gemini request. The visual agent
retains the user conversation and supplies the focused query for each tool call.
