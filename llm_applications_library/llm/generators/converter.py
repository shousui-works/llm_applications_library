import json
from typing import Any

from haystack import component


@component
class ProviderSelectableInstructGenerator:
    @component.output_types(response=str)
    def run(self, replies: list[str]):
        return {"response": replies[0]}


@component
class ProviderSelectableVisionGenerator:
    @component.output_types(response=str)
    def run(self, replies: dict[str, Any]):
        # DocumentVisionGeneratorは{'success': bool, 'content': str, 'usage': dict, 'error': str}を返す
        if replies.get("success", False):
            return {"response": replies["content"]}
        # エラーの場合はエラーメッセージを返す
        error_msg = replies.get("error", "Unknown error occurred")
        return {"response": f"Error: {error_msg}"}


@component
class Text2Json:
    @component.output_types(json_object=dict[str, Any])
    def run(self, text: str):
        print(f"text2json : {text}")
        return {"json_object": json.loads(text)}
