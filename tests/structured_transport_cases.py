"""实际 SDK 可消费的离线完成/流式响应。"""
import json

import httpx


def completion_response(value, request):
    base = {'id': 'offline', 'created': 0, 'model': request['model']}
    content = json.dumps(value)
    if not request.get('stream'):
        return httpx.Response(200, json={**base, 'object': 'chat.completion', 'choices': [
            {'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': content}}
        ]})
    events = [
        {**base, 'object': 'chat.completion.chunk', 'choices': [
            {'index': 0, 'delta': {'content': content[start:start + 97]}, 'finish_reason': None}
        ]} for start in range(0, len(content), 97)
    ]
    events.append({**base, 'object': 'chat.completion.chunk', 'choices': [
        {'index': 0, 'delta': {}, 'finish_reason': 'stop'}
    ]})
    body = ''.join('data: ' + json.dumps(event) + '\n\n' for event in events) + 'data: [DONE]\n\n'
    return httpx.Response(200, headers={'content-type': 'text/event-stream'}, content=body)
