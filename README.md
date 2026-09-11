# Vertex AI web search agent tool

This Dataiku plugin provides a custom agent tool that calls Gemini on Vertex AI with
Google Search grounding. It searches the live public web; it does **not** require or
query a Vertex AI Search datastore/application. It returns Gemini's answer, the web
search queries it generated, and the referenced source URLs.

## Configure

1. Install the plugin and create an **Agent Tool** of type **Vertex AI web search**.
2. Select a Dataiku Google/Vertex connection. Both `KEYPAIR` (the resolved service
   account JSON) and `OAUTH` (the resolved OAuth access token) are supported. The
   user who invokes the tool must be allowed to use that connection.
3. The tool uses the connection `projectId` and `location`/`region` by default. Set
   either override only when required, then choose a Gemini model that supports
   Google Search grounding.
4. Add the tool to a visual agent and describe when its knowledge base should be used.

Credentials are constructed from the resolved Dataiku connection at invocation time:
the service-account JSON for `KEYPAIR`, or a short-lived token for `OAUTH`. No
credential is stored in the tool configuration or returned to the agent.

The Google identity configured on the connection needs permission to search the
specified Vertex AI service. Vertex AI and the Google Search grounding feature must
be enabled for the chosen project, model, and region.

## Test payload

```json
{"input": {"query": "What are the latest James Webb Space Telescope updates?"}, "context": {}}
```

The tool is stateless: each call is a new grounded Gemini request. The visual agent
retains the user conversation and supplies the focused query for each tool call.

## Test outside Dataiku with ADC

The standalone test script uses Google Application Default Credentials (ADC), rather
than a Dataiku connection. Install its dependency, authenticate with Google Cloud,
and run it from the plugin root:

```bash
python3 -m pip install 'google-genai>=1.0.0,<2.0.0'
gcloud auth application-default login
python3 scripts/test_google_search_grounding.py \
  --project YOUR_GCP_PROJECT \
  "What are the most recent James Webb Space Telescope updates?"
```

For a service account, set `GOOGLE_APPLICATION_CREDENTIALS` to its JSON key file
instead of running the ADC login command.
