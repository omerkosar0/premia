"""
Premia — Flask Backend
======================
Serves the Premia dashboard and proxies Claude API calls server-side.
API key lives in .env — never exposed to the browser.

Usage:
  python3 app.py

Then open: http://localhost:5000
"""

import os
import json
from flask import Flask, request, Response, send_from_directory, jsonify
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = os.urandom(24)

# ── Anthropic client ──────────────────────────────────────
def get_client():
    key = os.getenv('ANTHROPIC_API_KEY')
    if not key:
        raise ValueError('ANTHROPIC_API_KEY not set in .env')
    return Anthropic(api_key=key)


# ── Routes ────────────────────────────────────────────────
@app.route('/')
def index():
    """Serve the main Premia dashboard."""
    return send_from_directory('static', 'index.html')


@app.route('/api/health')
def health():
    """Health check — confirms API key is configured."""
    key = os.getenv('ANTHROPIC_API_KEY')
    return jsonify({
        'status': 'ok',
        'api_key_configured': bool(key),
        'model': 'claude-sonnet-4-20250514'
    })


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Streaming chat endpoint.
    Receives: { system: str, messages: [...], max_tokens: int }
    Streams:  SSE events with text deltas
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON body'}), 400

        system   = data.get('system', '')
        messages = data.get('messages', [])
        max_tok  = data.get('max_tokens', 1024)

        if not messages:
            return jsonify({'error': 'No messages provided'}), 400

        client = get_client()

        def generate():
            try:
                with client.messages.stream(
                    model='claude-sonnet-4-20250514',
                    max_tokens=max_tok,
                    system=system,
                    messages=messages,
                ) as stream:
                    for text in stream.text_stream:
                        # SSE format
                        payload = json.dumps({'type': 'text', 'text': text})
                        yield f'data: {payload}\n\n'

                # Signal completion
                yield f'data: {json.dumps({"type": "done"})}\n\n'

            except Exception as e:
                err = json.dumps({'type': 'error', 'message': str(e)})
                yield f'data: {err}\n\n'

        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
                'Access-Control-Allow-Origin': '*',
            }
        )

    except ValueError as e:
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/api/chat', methods=['OPTIONS'])
def chat_options():
    """CORS preflight."""
    return Response(headers={
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
    })


# ── Dev server ────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'true').lower() == 'true'

    key = os.getenv('ANTHROPIC_API_KEY')
    print(f'\n{"="*52}')
    print(f'  PREMIA — Flask backend')
    print(f'{"="*52}')
    print(f'  URL      : http://localhost:{port}')
    print(f'  API key  : {"✓ configured" if key else "✗ NOT SET — add to .env"}')
    print(f'  Debug    : {debug}')
    print(f'{"="*52}\n')

    app.run(host='0.0.0.0', port=port, debug=debug)
