"""Gemini Google Search grounding agent tool, authenticated by Dataiku."""

import json

import dataiku
from dataiku.llm.agent_tools import BaseAgentTool


class VertexAIWebSearchTool(BaseAgentTool):
    def set_config(self, config, plugin_config):
        self.config = config or {}

    def get_descriptor(self, tool):
        return {
            "description": "Answers current public-web questions using Gemini with Google Search grounding. Returns an answer and cited web sources.",
            "inputSchema": {
                "title": "Grounded web-search query", "type": "object",
                "properties": {"query": {"type": "string", "description": "Question to answer using current web information."}},
                "required": ["query"],
            },
        }

    def invoke(self, input, trace):
        query = self._required(input.get("input", {}), "query")
        if trace:
            trace.span["name"] = "VERTEX_AI_GOOGLE_SEARCH_TOOL_CALL"
            trace.inputs["query"] = query
            trace.attributes["config"] = {key: self.config.get(key) for key in ("vertex_connection", "gcp_project_id", "location", "model", "temperature")}
        try:
            credentials, project, location = self._connection_credentials()
            response = self._ask_gemini(credentials, project, location, query)
            metadata = getattr((getattr(response, "candidates", None) or [None])[0], "grounding_metadata", None)
            searches = list(getattr(metadata, "web_search_queries", None) or [])
            snippets = {}
            for support in getattr(metadata, "grounding_supports", None) or []:
                text = getattr(getattr(support, "segment", None), "text", None)
                for index in getattr(support, "grounding_chunk_indices", None) or []:
                    if text:
                        snippets.setdefault(index, []).append(text)
            source_items = []
            for index, chunk in enumerate(getattr(metadata, "grounding_chunks", None) or []):
                web = getattr(chunk, "web", None)
                if web and web.uri:
                    title = web.title or web.uri
                    source_items.append({"type": "SIMPLE_DOCUMENT", "title": title, "url": web.uri,
                                         "textSnippet": " ".join(snippets.get(index, []))[:300] or title})
            if trace:
                trace.outputs.update({"source_count": len(source_items), "web_search_query_count": len(searches), "answer_length": len(response.text or "")})
            sources = [{
                "toolCallDescription": "Performed Gemini Google Search grounding for: {}".format(query),
                "items": source_items,
            }] if source_items else []
            return {
                "output": {"answer": response.text, "web_search_queries": searches, "sources": source_items},
                "sources": sources,
            }
        except Exception as error:
            if trace:
                trace.outputs["error"] = str(error)
            raise

    def load_sample_query(self, tool):
        return {"query": "What are the most recent James Webb Space Telescope updates this month?"}

    def _connection_credentials(self):
        from google.oauth2 import credentials, service_account

        name = self._required(self.config, "vertex_connection")
        try:
            info = dataiku.api_client().get_connection(name).get_info()
            params = info.get_resolved_params() or {}
        except Exception as error:
            raise RuntimeError("Unable to read Dataiku connection {!r}.".format(name)) from error
        connection_params = info.get_params() or {}
        project = self.config.get("gcp_project_id") or connection_params.get("project")
        location = (self.config.get("location") or connection_params.get("region"))
        if not project:
            raise ValueError("Set Google Cloud project ID or configure projectId on connection {!r}.".format(name))
        if not location:
            raise ValueError("Set Vertex AI location or configure location/region on connection {!r}.".format(name))
        if params.get("authType") == "KEYPAIR":
            key = params.get("appSecretContent") or params.get("keyPath")
            if not key:
                raise ValueError("No service-account key found in connection {!r}.".format(name))
            return service_account.Credentials.from_service_account_info(json.loads(key)), project, location
        if params.get("authType") == "OAUTH":
            oauth = params.get("resolvedOAuth2Credential") or info.get_oauth2_credential()
            if oauth and oauth.get("accessToken"):
                return credentials.Credentials(oauth["accessToken"]), project, location
            raise ValueError("No OAuth access token found in connection {!r}.".format(name))
        raise ValueError("Connection {!r} must use KEYPAIR or OAUTH authentication.".format(name))

    def _ask_gemini(self, credentials, project, location, query):
        from google import genai
        from google.genai import types

        client = genai.Client(
            vertexai=True, project=project, location=location,
            credentials=credentials, http_options=types.HttpOptions(api_version="v1"),
        )
        try:
            return client.models.generate_content(
                model=self.config.get("model", "gemini-2.5-flash"), contents=query,
                config=types.GenerateContentConfig(
                    temperature=float(self.config.get("temperature", 0.3)),
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                ),
            )
        finally:
            client.close()

    @staticmethod
    def _required(values, name):
        value = values.get(name)
        if not value or not str(value).strip():
            raise ValueError("Missing required tool configuration: {}".format(name))
        return str(value).strip()
